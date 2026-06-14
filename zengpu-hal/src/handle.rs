//! Generational-index slotmap and typed resource handles (plan §5 / D3).
//!
//! Every GPU resource is referenced by a small `Copy` handle that is an
//! `(index, generation)` pair into a device-owned [`SlotMap`]. A stale handle —
//! one whose generation no longer matches its slot — is rejected, which is what
//! gives the validation layer use-after-free detection without UB.

use core::marker::PhantomData;

/// A generational key into a [`SlotMap`]. `Copy`, backend-free, and carries a
/// phantom marker so a `BufferHandle` can't be used where a `TextureHandle` is
/// expected.
pub struct Handle<T> {
    idx: u32,
    generation: u32,
    _marker: PhantomData<fn() -> T>,
}

// Manual impls: deriving would add a `T: Trait` bound we don't want (the marker
// `T` is never actually stored).
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.generation == other.generation
    }
}
impl<T> Eq for Handle<T> {}
impl<T> core::hash::Hash for Handle<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.idx.hash(state);
        self.generation.hash(state);
    }
}
impl<T> core::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Handle({}, gen {})", self.idx, self.generation)
    }
}

impl<T> Handle<T> {
    /// The slot index. Doubles as the **bindless index** for resources placed in
    /// a descriptor table (plan D4).
    pub const fn index(self) -> u32 {
        self.idx
    }

    /// The generation this handle was minted with.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// A generational-index slotmap. Insertion reuses freed slots (bumping their
/// generation); a handle from a freed slot is detected as stale and rejected.
pub struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> SlotMap<T> {
    /// An empty slotmap.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    /// Insert a value, returning a fresh handle to it.
    pub fn insert(&mut self, value: T) -> Handle<T> {
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
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.slots
            .get(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_ref())
    }

    /// Mutably borrow the value for `handle`, or `None` if the handle is stale.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        self.slots
            .get_mut(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_mut())
    }

    /// Remove and return the value for `handle`. Bumps the slot's generation so
    /// any remaining copy of `handle` becomes stale. A stale handle removes
    /// nothing.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
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
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// Number of live values.
    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    /// Whether there are no live values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for SlotMap<T> {
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

    #[test]
    fn insert_get_remove() {
        let mut map: SlotMap<i32> = SlotMap::new();
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
        let mut map: SlotMap<i32> = SlotMap::new();
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
        let real: SlotMap<i32> = SlotMap::new();
        let mut other: SlotMap<i32> = SlotMap::new();
        let foreign = other.insert(1); // index 0 in a different map
        // `real` has no slot 0 — must not panic, must return None.
        assert_eq!(real.get(foreign), None);
    }
}
