//! Generational-index slotmap and typed resource handles.
//!
//! Every GPU resource is referenced by a small `Copy` handle: an `(index, generation)` pair into a
//! device-owned [`SlotMap`]. A stale handle, whose generation no longer matches its slot, is
//! rejected. That gives the validation layer use-after-free detection without UB.

use core::fmt::{Debug, Formatter, Result as FmtResult};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

/// Generational handle into a [`SlotMap`]. `Copy`, with no backend types.
///
/// The `K` tag keeps a `BufferHandle` from being used as a `TextureHandle`.
pub struct Handle<K> {
    idx: u32,
    generation: u32,
    _marker: PhantomData<fn() -> K>,
}

impl<K> Clone for Handle<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K> Copy for Handle<K> {}
impl<K> PartialEq for Handle<K> {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.generation == other.generation
    }
}
impl<K> Eq for Handle<K> {}
impl<K> Hash for Handle<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.idx.hash(state);
        self.generation.hash(state);
    }
}
impl<K> Debug for Handle<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "Handle({}, gen {})", self.idx, self.generation)
    }
}

impl<K> Handle<K> {
    /// Slot index, also the bindless index in a descriptor table.
    pub const fn index(self) -> u32 {
        self.idx
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

struct Slot<V> {
    generation: u32,
    value: Option<V>,
    reusable: bool,
}

/// Generational-index slotmap: typed [`Handle<K>`] keys, `V` values. Insertion reuses freed slots
/// with a bumped generation, so old handles are detected as stale.
///
/// `K` (a tag) and `V` (the value) are separate, so a backend can store its own buffer struct under
/// a public [`Handle<crate::marker::Buffer>`].
pub struct SlotMap<K, V> {
    slots: Vec<Slot<V>>,
    free: Vec<u32>,
    live: usize,
    _tag: PhantomData<fn() -> K>,
}

impl<K, V> SlotMap<K, V> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
            _tag: PhantomData,
        }
    }

    pub fn insert(&mut self, value: V) -> Handle<K> {
        if let Some(idx) = self.free.pop() {
            let slot = &mut self.slots[idx as usize];
            slot.value = Some(value);
            slot.reusable = false;
            self.live += 1;
            Handle {
                idx,
                generation: slot.generation,
                _marker: PhantomData,
            }
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                value: Some(value),
                reusable: false,
            });
            self.live += 1;
            Handle {
                idx,
                generation: 0,
                _marker: PhantomData,
            }
        }
    }

    /// Slot the next insert uses. Check it before publishing a handle to a fixed-size
    /// descriptor table.
    pub fn next_index(&self) -> u32 {
        self.free.last().copied().unwrap_or(self.slots.len() as u32)
    }

    pub fn get(&self, handle: Handle<K>) -> Option<&V> {
        self.slots
            .get(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_ref())
    }

    pub fn get_mut(&mut self, handle: Handle<K>) -> Option<&mut V> {
        self.slots
            .get_mut(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_mut())
    }

    /// Removes the value and bumps the generation, so copies of `handle` go stale.
    ///
    /// A stale handle removes nothing.
    pub fn remove(&mut self, handle: Handle<K>) -> Option<V> {
        let slot = self.slots.get_mut(handle.idx as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take();
        if value.is_some() {
            slot.generation = slot.generation.wrapping_add(1);
            slot.reusable = true;
            self.free.push(handle.idx);
            self.live -= 1;
        }
        value
    }

    /// Invalidates the handle but keeps the slot unusable while GPU work may still reference it.
    ///
    /// `release_retired` frees the index after completion.
    pub fn retire(&mut self, handle: Handle<K>) -> Option<V> {
        let slot = self.slots.get_mut(handle.idx as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take();
        if value.is_some() {
            slot.generation = slot.generation.wrapping_add(1);
            slot.reusable = false;
            self.live -= 1;
        }
        value
    }

    /// Returns `true` only for a retired slot. Live, free, or out-of-range slots return `false`,
    /// and the free list is untouched.
    pub fn release_retired(&mut self, index: u32) -> bool {
        let Some(slot) = self.slots.get_mut(index as usize) else {
            return false;
        };
        if slot.value.is_some() || slot.reusable {
            return false;
        }
        slot.reusable = true;
        self.free.push(index);
        true
    }

    pub fn contains(&self, handle: Handle<K>) -> bool {
        self.get(handle).is_some()
    }

    /// Returns the current generation of `index`, if it is in range.
    pub fn generation_at(&self, index: u32) -> Option<u32> {
        self.slots.get(index as usize).map(|slot| slot.generation)
    }

    /// Looks up by raw slot index, with no generation check.
    ///
    /// Used for bindless lookups, where `Bindings.buffers[i]` holds a slot index.
    pub fn get_by_slot_index(&self, idx: u32) -> Option<&V> {
        self.slots.get(idx as usize).and_then(|s| s.value.as_ref())
    }

    pub fn get_mut_by_slot_index(&mut self, idx: u32) -> Option<&mut V> {
        self.slots
            .get_mut(idx as usize)
            .and_then(|s| s.value.as_mut())
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove and yield all live values in slot order. Afterwards the map is empty and every old
    /// handle is stale. For deterministic cleanup in `Drop`.
    pub fn drain(&mut self) -> impl Iterator<Item = V> + '_ {
        self.free.clear();
        self.free.extend(0..self.slots.len() as u32);
        self.live = 0;
        self.slots.iter_mut().filter_map(|slot| {
            slot.reusable = true;
            let value = slot.value.take()?;
            slot.generation = slot.generation.wrapping_add(1);
            Some(value)
        })
    }
}

impl<K, V> Default for SlotMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

pub mod marker {
    pub enum Buffer {}
    pub enum Texture {}
    pub enum Sampler {}
    pub enum Shader {}
    pub enum Pipeline {}
    pub enum Surface {}
    pub enum RenderTarget {}
}

pub type BufferHandle = Handle<marker::Buffer>;
pub type TextureHandle = Handle<marker::Texture>;
pub type SamplerHandle = Handle<marker::Sampler>;
pub type ShaderHandle = Handle<marker::Shader>;
pub type PipelineHandle = Handle<marker::Pipeline>;
pub type SurfaceHandle = Handle<marker::Surface>;
pub type TargetHandle = Handle<marker::RenderTarget>;

#[cfg(test)]
mod tests {
    use super::*;

    type Map = SlotMap<marker::Buffer, i32>;

    #[test]
    fn insert_get_remove() {
        let mut map = Map::new();
        assert!(map.is_empty());
        let h = map.insert(42);
        assert_eq!(map.get(h), Some(&42));
        assert_eq!(map.len(), 1);
        *map.get_mut(h).unwrap() = 7;
        assert_eq!(map.get(h), Some(&7));
        assert_eq!(map.remove(h), Some(7));
        assert!(map.is_empty());
    }

    #[test]
    fn next_index_tracks_reused_slots() {
        let mut map = Map::new();
        assert_eq!(map.next_index(), 0);
        let first = map.insert(10);
        assert_eq!(map.next_index(), 1);
        map.remove(first);
        assert_eq!(map.next_index(), first.index());
    }

    #[test]
    fn retired_slot_is_stale_but_not_reused_until_release() {
        let mut map = Map::new();
        let retired = map.insert(10);
        assert_eq!(map.retire(retired), Some(10));
        assert_eq!(map.get(retired), None);
        assert!(map.is_empty());

        let second = map.insert(20);
        assert_ne!(second.index(), retired.index());
        assert!(!map.release_retired(second.index()));
        assert!(map.release_retired(retired.index()));
        assert!(!map.release_retired(retired.index()));

        let reused = map.insert(30);
        assert_eq!(reused.index(), retired.index());
        assert_ne!(reused.generation(), retired.generation());
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn stale_handle_after_remove_is_rejected() {
        let mut map = Map::new();
        let h = map.insert(10);
        assert_eq!(map.remove(h), Some(10));
        assert_eq!(map.get(h), None);
        assert!(!map.contains(h));
        assert_eq!(map.remove(h), None);

        let h2 = map.insert(20);
        assert_eq!(h2.index(), h.index());
        assert_ne!(h2.generation(), h.generation());
        assert_eq!(map.get(h), None, "old handle must stay stale");
        assert_eq!(map.get(h2), Some(&20));
    }

    #[test]
    fn out_of_range_handle_is_safe() {
        let real = Map::new();
        let mut other = Map::new();
        let foreign = other.insert(1);
        assert_eq!(real.get(foreign), None);
    }

    #[test]
    fn generation_at_tracks_slot() {
        let mut map = Map::new();
        let h = map.insert(1);
        assert_eq!(map.generation_at(h.index()), Some(0));
        map.remove(h);
        assert_eq!(map.generation_at(h.index()), Some(1));
        assert_eq!(map.generation_at(999), None);
    }

    #[test]
    fn drain_removes_all_and_invalidates_handles() {
        let mut map = Map::new();
        let h0 = map.insert(10);
        let h1 = map.insert(20);
        map.remove(h0);
        let drained: Vec<_> = map.drain().collect();
        assert_eq!(drained, vec![20]);
        assert!(map.is_empty());
        assert_eq!(map.get(h1), None);
    }
}
