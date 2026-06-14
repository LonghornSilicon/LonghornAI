// LonghornAI — pluggable allocator interface and built-in implementations.
//
// Kernels never see allocators directly; only `Tensor` does. The interface is
// deliberately small (alloc / free / reset) so a future paged or NUMA-aware
// allocator can drop in without disturbing call sites.
#ifndef LONGHORNAI_CORE_ALLOCATOR_HPP
#define LONGHORNAI_CORE_ALLOCATOR_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace lh {

class IAllocator {
public:
    virtual ~IAllocator() = default;
    virtual void* alloc(size_t bytes, size_t align) = 0;
    virtual void free(void* ptr) = 0;
    virtual void reset() {}  // no-op for non-arena allocators
    virtual const char* name() const = 0;
};

// Default heap allocator: aligned `operator new` / `operator delete`.
class HeapAllocator final : public IAllocator {
public:
    void* alloc(size_t bytes, size_t align) override {
        if (bytes == 0) return nullptr;
        if (align < alignof(std::max_align_t)) align = alignof(std::max_align_t);
        // Pad to a multiple of `align` (a requirement of aligned new on some
        // platforms) so the call is well-defined for any byte count.
        const size_t padded = (bytes + align - 1) & ~(align - 1);
        return ::operator new(padded, std::align_val_t{align});
    }

    void free(void* ptr) override {
        if (!ptr) return;
        ::operator delete(ptr, std::align_val_t{alignof(std::max_align_t)});
    }

    const char* name() const override { return "heap"; }

    static HeapAllocator& instance() {
        static HeapAllocator a;
        return a;
    }
};

// Bump-pointer arena. Allocations are freed in bulk by `reset()`. Used by the
// per-kernel scratchpad in tests and the bench harness.
class ArenaAllocator final : public IAllocator {
public:
    explicit ArenaAllocator(size_t capacity_bytes)
        : buffer_(static_cast<uint8_t*>(
              ::operator new(capacity_bytes,
                             std::align_val_t{alignof(std::max_align_t)}))),
          capacity_(capacity_bytes), offset_(0) {}

    ~ArenaAllocator() override {
        ::operator delete(buffer_, std::align_val_t{alignof(std::max_align_t)});
    }

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    void* alloc(size_t bytes, size_t align) override {
        if (bytes == 0) return nullptr;
        const size_t aligned = (offset_ + align - 1) & ~(align - 1);
        if (aligned + bytes > capacity_) return nullptr;  // OOM in this arena
        void* p = buffer_ + aligned;
        offset_ = aligned + bytes;
        return p;
    }

    void free(void* /*ptr*/) override {}  // bulk-freed in `reset`

    void reset() override { offset_ = 0; }

    size_t used() const { return offset_; }
    size_t capacity() const { return capacity_; }

    const char* name() const override { return "arena"; }

private:
    uint8_t* buffer_;
    size_t capacity_;
    size_t offset_;
};

// Page-aligned allocator. The KV cache will eventually demand 4 KiB / 2 MiB
// alignment; this is the seat for that policy. Today it just over-aligns to
// 4 KiB so call sites can be written against it without changing later.
class PageAlignedAllocator final : public IAllocator {
public:
    static constexpr size_t kPage = 4096;

    void* alloc(size_t bytes, size_t /*align*/) override {
        if (bytes == 0) return nullptr;
        const size_t padded = (bytes + kPage - 1) & ~(kPage - 1);
        return ::operator new(padded, std::align_val_t{kPage});
    }

    void free(void* ptr) override {
        if (!ptr) return;
        ::operator delete(ptr, std::align_val_t{kPage});
    }

    const char* name() const override { return "page"; }

    static PageAlignedAllocator& instance() {
        static PageAlignedAllocator a;
        return a;
    }
};

}  // namespace lh

#endif  // LONGHORNAI_CORE_ALLOCATOR_HPP
