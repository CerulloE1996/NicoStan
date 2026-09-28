#ifdef __linux__
  // Discard complete interior pages; malloc metadata and boundary
  // pages remain intact. Call only for unused, still-allocated storage.
  static size_t nicostan_discard_pages(void* pointer, size_t bytes) {
    if (pointer == nullptr || bytes == 0) return 0;
    const long page_value = sysconf(_SC_PAGESIZE);
    if (page_value <= 0) throw std::runtime_error("Page size unavailable");
    const uintptr_t page = static_cast<uintptr_t>(page_value);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(pointer);
    const uintptr_t end = begin + bytes;
    if (end < begin) throw std::runtime_error("Discard range overflow");
    const uintptr_t first = begin + ((page - (begin % page)) % page);
    const uintptr_t last = end - (end % page);
    if (first >= last) return 0;
    if (madvise(reinterpret_cast<void*>(first), last - first, MADV_DONTNEED) != 0)
      throw std::runtime_error("MADV_DONTNEED failed");
    return last - first;
  }

  size_t nicostan_discard_unused_pages() {
    if (!nested_cur_blocks_.empty() || !nested_next_locs_.empty()
        || !nested_cur_block_ends_.empty() || cur_block_ != 0
        || next_loc_ != blocks_[0])
      throw std::runtime_error("Cannot discard a live Stan arena");
    size_t bytes = 0;
    for (size_t i = 0; i < blocks_.size(); ++i)
      bytes += nicostan_discard_pages(blocks_[i], sizes_[i]);
    return bytes;
  }
#endif
