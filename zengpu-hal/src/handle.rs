//! Generational-index slotmap and typed resource handles.
//!
//! Every GPU resource is referenced by a small `Copy` handle that is an
//! `(owner, index, generation)` key into a device-owned [`SlotMap`]. A stale handle —
//! one whose generation no longer matches its slot — is rejected, which is what
//! gives the validation layer use-after-free detection without UB.

use core::fmt::{Debug, Formatter, Result as FmtResult};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};

/// A generational key into a [`SlotMap`]. `Copy`, backend-free, and carries a
/// phantom marker `K` so a `BufferHandle` can't be used where a `TextureHandle`
/// is expected.
pub struct Handle<K> {
    owner: u64,
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
        self.owner == other.owner && self.idx == other.idx && self.generation == other.generation
    }
}
impl<K> Eq for Handle<K> {}
impl<K> Hash for Handle<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.owner.hash(state);
        self.idx.hash(state);
        self.generation.hash(state);
    }
}
impl<K> Debug for Handle<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Handle({}, gen {}, owner {})",
            self.idx, self.generation, self.owner
        )
    }
}

impl<K> Handle<K> {
    /// Identity of the resource table that minted this handle.
    pub const fn owner(self) -> u64 {
        self.owner
    }

    /// The slot index. Doubles as the **bindless index** for resources placed in
    /// a descriptor table.
    pub const fn index(self) -> u32 {
        self.idx
    }

    /// The generation this handle was minted with.
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// An unowned handle, useful as an invalid sentinel. Resource tables reject
    /// it even when its index and generation match a live resource. To restore
    /// a live handle, preserve its owner with [`Self::from_owned_parts`].
    pub const fn from_parts(index: u32, generation: u32) -> Self {
        Self::from_owned_parts(index, generation, 0)
    }

    /// Restore all parts of a handle, including its resource-table identity.
    pub const fn from_owned_parts(index: u32, generation: u32, owner: u64) -> Self {
        Self {
            owner,
            idx: index,
            generation,
            _marker: PhantomData,
        }
    }
}

struct Slot<V> {
    generation: u32,
    value: Option<V>,
    reusable: bool,
}

/// A generational-index slotmap: keys are typed [`Handle<K>`], values are `V`.
/// Insertion reuses freed slots (bumping their generation); a handle from a
/// freed slot or a different table is detected as stale and rejected. Slots
/// whose generations are exhausted are never reused.
///
/// The key tag `K` and the stored value `V` are separate, so a backend can store
/// its own buffer struct under a public [`Handle<crate::marker::Buffer>`].
pub struct SlotMap<K, V> {
    owner: u64,
    slots: Vec<Slot<V>>,
    free: Vec<u32>,
    live: usize,
    _tag: PhantomData<fn() -> K>,
}

impl<K, V> SlotMap<K, V> {
    /// An empty slotmap.
    pub fn new() -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        // try_update requires a newer Rust than our MSRV.
        #[allow(deprecated)]
        let owner = match NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        {
            Ok(owner) => owner,
            Err(_) => panic!("GPU resource-table identity exhausted"),
        };
        Self {
            owner,
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
            _tag: PhantomData,
        }
    }

    /// Insert a value, returning a fresh handle to it.
    pub fn insert(&mut self, value: V) -> Handle<K> {
        if let Some(idx) = self.free.pop() {
            let slot = &mut self.slots[idx as usize];
            slot.value = Some(value);
            slot.reusable = false;
            self.live += 1;
            Handle {
                owner: self.owner,
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
                owner: self.owner,
                idx,
                generation: 0,
                _marker: PhantomData,
            }
        }
    }

    /// Slot index the next insertion will use. Useful when a backend has a
    /// fixed-size bindless descriptor table and must reject an insertion
    /// before publishing an out-of-range handle.
    pub fn next_index(&self) -> u32 {
        self.free.last().copied().unwrap_or(self.slots.len() as u32)
    }

    /// Borrow the value for `handle`, or `None` if the handle is stale.
    pub fn get(&self, handle: Handle<K>) -> Option<&V> {
        if handle.owner != self.owner {
            return None;
        }
        self.slots
            .get(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_ref())
    }

    /// Mutably borrow the value for `handle`, or `None` if the handle is stale.
    pub fn get_mut(&mut self, handle: Handle<K>) -> Option<&mut V> {
        if handle.owner != self.owner {
            return None;
        }
        self.slots
            .get_mut(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_mut())
    }

    /// Remove and return the value for `handle`. Bumps the slot's generation so
    /// any remaining copy of `handle` becomes stale. A stale handle removes
    /// nothing.
    pub fn remove(&mut self, handle: Handle<K>) -> Option<V> {
        if handle.owner != self.owner {
            return None;
        }
        let slot = self.slots.get_mut(handle.idx as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take();
        if value.is_some() {
            slot.generation = slot.generation.saturating_add(1);
            slot.reusable = slot.generation < u32::MAX;
            if slot.reusable {
                self.free.push(handle.idx);
            }
            self.live -= 1;
        }
        value
    }

    /// Remove a value and invalidate its handle without making the slot
    /// reusable. Backends use this when destruction is requested while GPU
    /// work may still reference the slot or its bindless descriptor.
    /// [`Self::release_retired`] makes the index reusable after completion.
    pub fn retire(&mut self, handle: Handle<K>) -> Option<V> {
        if handle.owner != self.owner {
            return None;
        }
        let slot = self.slots.get_mut(handle.idx as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take();
        if value.is_some() {
            slot.generation = slot.generation.saturating_add(1);
            slot.reusable = false;
            self.live -= 1;
        }
        value
    }

    /// Release a previously retired slot for reuse. Returns `true` only for a
    /// valid retired slot; live, already-free, and out-of-range slots return
    /// `false` and do not alter the free list.
    pub fn release_retired(&mut self, index: u32) -> bool {
        let Some(slot) = self.slots.get_mut(index as usize) else {
            return false;
        };
        if slot.value.is_some() || slot.reusable || slot.generation == u32::MAX {
            return false;
        }
        slot.reusable = true;
        self.free.push(index);
        true
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
        self.live
    }

    /// Whether there are no live values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove and yield all live values in slot order.  After draining, the
    /// map is empty and all previously valid handles are stale.  Useful for
    /// deterministic cleanup in `Drop` implementations.
    /// Removal is eager, even if the returned iterator is never consumed.
    pub fn drain(&mut self) -> impl Iterator<Item = V> + '_ {
        self.free.clear();
        self.live = 0;
        let values: Vec<_> = self
            .slots
            .iter_mut()
            .filter_map(|slot| {
                let value = slot.value.take();
                if value.is_some() {
                    slot.generation = slot.generation.saturating_add(1);
                }
                slot.reusable = slot.generation < u32::MAX;
                value
            })
            .collect();
        self.free.extend(
            self.slots
                .iter()
                .enumerate()
                .filter_map(|(index, slot)| slot.reusable.then_some(index as u32)),
        );
        values.into_iter()
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
    /// Marker for query-pool handles.
    pub enum QueryPool {}
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
/// Handle to a GPU query pool.
pub type QueryPoolHandle = Handle<marker::QueryPool>;

#[cfg(test)]
mod tests {
    use super::*;

    // Any tag works for the tests; reuse the buffer marker.
    type Map = SlotMap<marker::Buffer, i32>;

    #[test]
    fn populated_foreign_maps_and_recreated_maps_reject_handles() {
        let mut first = Map::new();
        let old = first.insert(10);
        let mut second = Map::new();
        let other = second.insert(20);
        assert_eq!(old.index(), other.index());
        assert_eq!(old.generation(), other.generation());
        assert_ne!(old, other);
        assert_eq!(second.get(old), None);
        assert_eq!(second.get_mut(old), None);
        assert_eq!(second.remove(old), None);
        assert_eq!(second.retire(old), None);
        assert_eq!(second.get(other), Some(&20));
        assert_eq!(
            second.get(Handle::from_parts(other.index(), other.generation())),
            None
        );
        drop(first);
        let mut recreated = Map::new();
        recreated.insert(30);
        assert_eq!(recreated.get(old), None);
    }

    #[test]
    fn dropping_partially_consumed_drain_removes_every_value() {
        let mut map = Map::new();
        let first = map.insert(10);
        let second = map.insert(20);
        let mut drained = map.drain();
        assert_eq!(drained.next(), Some(10));
        drop(drained);
        assert!(map.is_empty());
        assert_eq!(map.get(first), None);
        assert_eq!(map.get(second), None);
        assert_eq!(map.get_by_slot_index(second.index()), None);
        let reused = map.insert(30);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(reused), Some(&30));
        drop(map.drain());
        assert_eq!(map.get_by_slot_index(reused.index()), None);
    }

    #[test]
    fn exhausted_generations_are_not_reused() {
        let mut map = Map::new();
        let original = map.insert(10);
        map.slots[0].generation = u32::MAX - 1;
        let last = Handle::from_owned_parts(0, u32::MAX - 1, original.owner());
        assert_eq!(map.remove(last), Some(10));
        assert!(!map.release_retired(0));
        assert_ne!(map.insert(20).index(), 0);
        drop(map.drain());
        assert_ne!(map.insert(30).index(), 0);
        assert_eq!(map.get(original), None);
    }

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
    fn rebuilt_handles_are_checked() {
        let mut map = Map::new();
        let h = map.insert(5);
        let same = Handle::from_owned_parts(h.index(), h.generation(), h.owner());
        assert_eq!(map.get(same), Some(&5));
        map.remove(h);
        map.insert(6);
        assert_eq!(map.get(same), None);
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
