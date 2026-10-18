// IREE executable library ABI v0.6, pinned to iree-base-compiler 3.12.0.
// Layouts from iree/hal/local/executable_library.h (Apache-2.0 WITH LLVM-exception).
#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>

typedef struct { uint32_t version; const char* name; uint32_t features; int sanitizer; } library_header;
typedef struct {
  const uint32_t* constants;
  int (*import_thunk)(void);
  const void* import_funcs;
  const void* import_contexts;
  uint64_t processor[8];
} environment;
typedef struct {
  uint32_t size_x, size_y;
  uint16_t size_z, constant_count;
  uint32_t count_x, count_y;
  uint16_t count_z;
  uint8_t max_concurrency, binding_count;
  const uint32_t* constants;
  void* const* bindings;
  const size_t* lengths;
} dispatch_state;
typedef struct {
  uint32_t x, y;
  uint16_t z, reserved;
  uint32_t processor_id;
  void* local_memory;
  uint32_t local_memory_size;
} workgroup_state;
typedef int (*dispatch_fn)(const environment*, const dispatch_state*, const workgroup_state*);
typedef struct {
  uint64_t flags;
  uint16_t local_memory_pages;
  uint8_t constant_count, binding_count;
  uint32_t size_x, size_y;
  uint16_t size_z, parameter_count;
  uint64_t reserved[5];
} dispatch_attrs;
typedef struct {
  const library_header* header;
  struct { uint32_t count; const char* const* symbols; } imports;
  struct {
    uint32_t count;
    const dispatch_fn* functions;
    const dispatch_attrs* attrs;
    const void* params;
    const void* occupancy;
    const char* const* names;
    const void* tags;
    const void* parameter_names;
    const void* source_locations;
    const void* stage_locations;
  } exports;
  struct { uint32_t count; } constants;
  struct { uint32_t count; const void* files; } sources;
} executable;
