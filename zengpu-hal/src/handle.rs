//! Generational-index slotmap and typed resource handles.
//!
//! Every GPU resource is a small `Copy` handle: an `(index, generation)` pair into a device-owned
//! [`SlotMap`]. A stale handle is rejected, which gives the validation layer use-after-free
//! detection without UB.

use core::marker::PhantomData;

/// Generational key into a [`SlotMap`]. `Copy`, backend-free. The phantom marker keeps a
/// `BufferHandle` from being used as a `TextureHandle`.
pub struct Handle<T> {
    idx: u32,
    generation: u32,
    _marker: PhantomData<fn() -> T>,
}

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
    /// The slot index. Also the bindless index for resources placed in a descriptor table.
    pub const fn index(self) -> u32 {
        self.idx
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Reused slots bump their generation, so a handle to a freed slot is rejected as stale.
pub struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> SlotMap<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

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

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.slots
            .get(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_ref())
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        self.slots
            .get_mut(handle.idx as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.value.as_mut())
    }

    /// Removes the value and bumps the generation, so copies of `handle` go stale.
    ///
    /// A stale handle removes nothing.
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

    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// Current generation of the slot at `index`, if in range. For stale-handle diagnostics.
    pub fn generation_at(&self, index: u32) -> Option<u32> {
        self.slots.get(index as usize).map(|slot| slot.generation)
    }

    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for SlotMap<T> {
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
        let real: SlotMap<i32> = SlotMap::new();
        let mut other: SlotMap<i32> = SlotMap::new();
        let foreign = other.insert(1);
        assert_eq!(real.get(foreign), None);
    }

    #[test]
    fn generation_at_tracks_slot() {
        let mut map: SlotMap<i32> = SlotMap::new();
        let h = map.insert(1);
        assert_eq!(map.generation_at(h.index()), Some(0));
        map.remove(h);
        assert_eq!(map.generation_at(h.index()), Some(1));
        assert_eq!(map.generation_at(999), None);
    }
}
