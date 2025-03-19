# Memory paging

## Introduction

- The memory management in Linux is a complex system that evolved over the years and included more and more
  functionality to support a variety of systems from MMU-less microcontrollers to supercomputers.

- The physical memory is not necessarily contiguous; it
  might be accessible as a set of distinct address ranges. Besides, different CPU architectures, and even different
  implementations of the same architecture have different views of how these address ranges are defined.

- All this makes dealing directly with physical memory quite complex and to avoid this complexity a concept of virtual
  memory was developed. The virtual memory abstracts the details of physical memory from the application software,
  allows to keep only needed information in the physical memory (demand paging) and provides a mechanism for the
  protection and controlled sharing of data between processes.

- With virtual memory, each and every memory access uses a virtual address. When the CPU decodes an instruction that
  reads (or writes) from (or to) the system memory, it translates the virtual address encoded in that instruction to a
  physical address that the memory controller can understand.

- The physical system memory is divided into page frames, or pages. The size of each page is architecture specific. Some
  architectures allow selection of the page size from several supported values; this selection is performed at the
  kernel
  build time by setting an appropriate kernel configuration option.

## Page Tables

- Page tables map virtual addresses as seen by the CPU into physical addresses as seen on the external memory bus.

- Linux defines page tables as a hierarchy which is currently five levels in height. The architecture code for each
  supported architecture will then map this to the restrictions of the hardware.

## References

- https://www.kernel.org/doc/html/v6.4/admin-guide/mm/concepts.html
- https://docs.kernel.org/mm/page_tables.html