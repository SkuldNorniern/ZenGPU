//! Generational-index slotmap and typed resource handles.
//!
//! Every GPU resource is referenced by a small `Copy` handle that is an
//! `(index, generation)` pair into a device-owned [`SlotMap`]. A stale handle —
//! one whose generation no longer matches its slot — is rejected, which is what
//! gives the validation layer use-after-free detection without UB.

use core::fmt::{Debug, Formatter, Result as FmtResult};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

/// A generational key into a [`SlotMap`]. `Copy`, backend-free, and carries a
/// phantom marker `K` so a `BufferHandle` can't be used where a `TextureHandle`
/// is expected.
pub struct Handle<K> {
    idx: u32,
    generation: u32,
    _marker: PhantomData<fn() -> K>,
}

// Manual impls: deriving would add a `K: Trait` bound we don't want (the marker
// `K` is never actually stored).
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
    /// The slot index. Doubles as the **bindless index** for resources placed in
    /// a descriptor table.
    pub const fn index(self) -> u32 {
        self.idx
    }

    /// The generation this handle was minted with.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

struct Slot<V> {
    generation: u32,
    value: Option<V>,
}

/// A generational-index slotmap: keys are typed [`Handle<K>`], values are `V`.
/// Insertion reuses freed slots (bumping their generation); a handle from a
/// freed slot is detected as stale and rejected.
///
/// The key tag `K` and the stored value `V` are separate, so a backend can store
/// its own buffer struct under a public [`Handle<crate::marker::Buffer>`].
pub struct SlotMap<K, V> {
    slots: Vec<Slot<V>>,
    free: Vec<u32>,
    _tag: PhantomData<fn() -> K>,
}

impl<K, V> SlotMap<K, V> {
    /// An empty slotmap.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            _tag: PhantomData,
        }
    }

    /// Insert a value, returning a fresh handle to it.
    pub fn insert(&mut self, value: V) -> Handle<K> {
        if let Some(idx) = self.free.pop() {
            let slot = &mut self.slots[idx as usize];
            slot.value = Some(value);
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
            });
            Handle {
                idx,
                generation: 0,
                _marker: PhantomData,
            }
        }
    }

    /// Borrow the value for `handle`, or `None` if the handle is stale.
    pub fn get(&self, handle: Handle<K>) -> Option<&V> {
        self.slots
            .get(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_ref())
    }

    /// Mutably borrow the value for `handle`, or `None` if the handle is stale.
    pub fn get_mut(&mut self, handle: Handle<K>) -> Option<&mut V> {
        self.slots
            .get_mut(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_mut())
    }

    /// Remove and return the value for `handle`. Bumps the slot's generation so
    /// any remaining copy of `handle` becomes stale. A stale handle removes
    /// nothing.
    pub fn remove(&mut self, handle: Handle<K>) -> Option<V> {
        let slot = self.slots.get_mut(handle.idx as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take();
        if value.is_some() {
            slot.generation = slot.generation.wrapping_add(1);
            self.free.push(handle.idx);
        }
        value
    }

    /// Whether `handle` still refers to a live value.
    pub fn contains(&self, handle: Handle<K>) -> bool {
        self.get(handle).is_some()
    }

    /// The current generation of the slot at `index`, if the index is in range.
    /// Lets a backend build a precise stale-handle diagnostic.
    pub fn generation_at(&self, index: u32) -> Option<u32> {
        self.slots.get(index as usize).map(|slot| slot.generation)
    }

    /// Get a live value by raw slot index, bypassing the generation check.
    /// Intended for bindless-index lookups: `Bindings.buffers[i]` is a slot
    /// index, not a full generational handle.
    pub fn get_by_slot_index(&self, idx: u32) -> Option<&V> {
        self.slots.get(idx as usize).and_then(|s| s.value.as_ref())
    }

    /// Mutably get a live value by raw slot index.
    pub fn get_mut_by_slot_index(&mut self, idx: u32) -> Option<&mut V> {
        self.slots
            .get_mut(idx as usize)
            .and_then(|s| s.value.as_mut())
    }

    /// Number of live values.
    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    /// Whether there are no live values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove and yield all live values in slot order.  After draining, the
    /// map is empty and all previously valid handles are stale.  Useful for
    /// deterministic cleanup in `Drop` implementations.
    pub fn drain(&mut self) -> impl Iterator<Item = V> + '_ {
        // Reset free list to cover every slot so len() == 0 after draining.
        self.free.clear();
        self.free.extend(0..self.slots.len() as u32);
        self.slots.iter_mut().filter_map(|slot| {
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

/// Zero-sized marker types that tag a [`Handle`]. Never constructed.
pub mod marker {
    /// Marker for buffer handles.
    pub enum Buffer {}
    /// Marker for texture handles.
    pub enum Texture {}
    /// Marker for sampler handles.
    pub enum Sampler {}
    /// Marker for shader-module handles.
    pub enum Shader {}
    /// Marker for pipeline handles.
    pub enum Pipeline {}
    /// Marker for surface handles.
    pub enum Surface {}
    /// Marker for render-target handles.
    pub enum RenderTarget {}
}

/// Handle to a GPU buffer.
pub type BufferHandle = Handle<marker::Buffer>;
/// Handle to a GPU texture.
pub type TextureHandle = Handle<marker::Texture>;
/// Handle to a sampler.
pub type SamplerHandle = Handle<marker::Sampler>;
/// Handle to a shader module.
pub type ShaderHandle = Handle<marker::Shader>;
/// Handle to a compute or graphics pipeline.
pub type PipelineHandle = Handle<marker::Pipeline>;
/// Handle to a presentable surface.
pub type SurfaceHandle = Handle<marker::Surface>;
/// Handle to a render target (swapchain image or offscreen texture).
pub type TargetHandle = Handle<marker::RenderTarget>;

#[cfg(test)]
mod tests {
    use super::*;

    // Any tag works for the tests; reuse the buffer marker.
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
    fn stale_handle_after_remove_is_rejected() {
        let mut map = Map::new();
        let h = map.insert(10);
        assert_eq!(map.remove(h), Some(10));
        // Generation bumped → the old handle is now stale.
        assert_eq!(map.get(h), None);
        assert!(!map.contains(h));
        assert_eq!(map.remove(h), None);

        // Reusing the slot keeps the index but advances the generation.
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
        let foreign = other.insert(1); // index 0 in a different map
        // `real` has no slot 0 — must not panic, must return None.
        assert_eq!(real.get(foreign), None);
    }

    #[test]
    fn generation_at_tracks_slot() {
        let mut map = Map::new();
        let h = map.insert(1);
        assert_eq!(map.generation_at(h.index()), Some(0));
        map.remove(h);
        assert_eq!(map.generation_at(h.index()), Some(1)); // bumped on free
        assert_eq!(map.generation_at(999), None); // out of range
    }

    #[test]
    fn drain_removes_all_and_invalidates_handles() {
        let mut map = Map::new();
        let h0 = map.insert(10);
        let h1 = map.insert(20);
        map.remove(h0); // h0 slot is free before drain
        let drained: Vec<_> = map.drain().collect();
        assert_eq!(drained, vec![20]); // only h1 was live
        assert!(map.is_empty());
        assert_eq!(map.get(h1), None); // h1 is now stale
    }
}
