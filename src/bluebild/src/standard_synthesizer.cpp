#include <complex>
#include <optional>
#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "bluebild/standard_synthesizer.hpp"
#include "host/ss_host.hpp"
#include "host/standard_synthesizer_host.hpp"
#include "context_internal.hpp"
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "memory/buffer.hpp"
#include "gpu/standard_synthesizer_gpu.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#endif

namespace bluebild {

template <typename T> struct SSInternal {
  SSInternal(std::shared_ptr<ContextInternal> ctx, const T wl,
             const size_t Nl, const size_t Nh, const size_t Nw,
             const T* pix_grid, T* stats_std_cum, T* stats_lsq_cum)
      : wl_(wl), Nl_(Nl), Nh_(Nh), Nw_(Nw), pix_grid_(std::move(pix_grid)),
        stats_std_cum_(stats_std_cum), stats_lsq_cum_(stats_lsq_cum),
        ctx_(std::move(ctx)) {
    if (ctx_->processing_unit() == BLUEBILD_PU_GPU) {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)

        assert(is_device_ptr(pix_grid)      == false);
        assert(is_device_ptr(stats_std_cum) == false);
        assert(is_device_ptr(stats_lsq_cum) == false);
        
        pix_grid_size_ = 3 * Nh * Nw;
        pix_grid_buff_ = create_buffer<T>(ctx_->allocators().gpu(), pix_grid_size_);
        gpu::check_status(gpu::memcpy_async(pix_grid_buff_.get(), pix_grid, pix_grid_size_ * sizeof(T),
                                            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
#else
      throw GPUSupportError();
#endif
    }
  }

    void exec(const T* d, const std::complex<T>* v, const T* xyz, const std::complex<T>* w,
              const size_t* c_idx, const size_t Na, const size_t Ne, const size_t Nb, const bool d2h) {
    if (ctx_->processing_unit() == BLUEBILD_PU_CPU) {
        standard_synthesizer_host<T>(*ctx_.get(), d, v, xyz, w, c_idx, Nl_, pix_grid_, wl_,
                                     Na, Nb, 3, Ne, Nh_, Nw_, stats_std_cum_, stats_lsq_cum_);
    } else {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
        
      //EO: to check what internal mem maps are containing at start of new iteration
      //ctx_->allocators().gpu()->print();

      assert(is_device_ptr(xyz) == false);
      assert(is_device_ptr(w)   == false);
        
      size_t d_size    = Ne;
      size_t v_size    = Nb * Ne;
      size_t xyz_size  = Na * 3;
      size_t w_size    = Na * Nb;
      size_t cidx_size = Ne;

      BufferType<T> d_buff   = create_buffer<T>(ctx_->allocators().gpu(), d_size);
      BufferType<T> xyz_buff = create_buffer<T>(ctx_->allocators().gpu(), xyz_size);
      BufferType<std::size_t> cidx_buff  = create_buffer<std::size_t>(ctx_->allocators().gpu(), cidx_size);
      BufferType<std::size_t> cinfo_buff = create_buffer<std::size_t>(ctx_->allocators().gpu(), 2 * cidx_size);
      BufferType<gpu::ComplexType<T>> v_buff = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(), v_size);
      BufferType<gpu::ComplexType<T>> w_buff = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(), w_size);
      
      gpu::check_status(gpu::memcpy_async(d_buff.get(), d, d_size * sizeof(T),
                                          gpu::flag::MemcpyHostToDevice,
                                          ctx_->gpu_stream()));
      gpu::check_status(gpu::memcpy_async(v_buff.get(), v, v_size * sizeof(gpu::ComplexType<T>),
                                          gpu::flag::MemcpyHostToDevice,
                                          ctx_->gpu_stream()));
      auto bufferXYZ = create_buffer<T>(ctx_->allocators().host(), xyz_size);
      mean_center(bufferXYZ.get(), xyz, Na);
      gpu::check_status(gpu::memcpy_async(xyz_buff.get(), bufferXYZ.get(), xyz_size * sizeof(T),
                                          gpu::flag::MemcpyHostToDevice,
                                          ctx_->gpu_stream()));
      gpu::check_status(gpu::memcpy_async(w_buff.get(), w, w_size * sizeof(gpu::ComplexType<T>),
                                          gpu::flag::MemcpyHostToDevice,
                                          ctx_->gpu_stream()));

      gpu::check_status(gpu::memcpy_async(cidx_buff.get(), c_idx, cidx_size * sizeof(std::size_t),
                                          gpu::flag::MemcpyHostToDevice,
                                          ctx_->gpu_stream()));

      // Compute layers' info (start + thickness)
      size_t offset = 0;
      size_t *c_info = new size_t[Nl_*2]{0};
      for (size_t i=0; i<Ne; i++) c_info[Nl_ + c_idx[i]] += 1;
      for (size_t i=0; i<Nl_; i++) {
          c_info[i] = offset;
          offset   += c_info[Nl_ + i];
      }

      gpu::check_status(gpu::memcpy_async(cinfo_buff.get(), c_info, 2 * cidx_size * sizeof(std::size_t),
                                          gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));

      // First epoch: if not enough memory on device, we split the processing along Nw
      //              (can't be done at initialization due to lack of information)
      if (first_epoch_) {

          size_t free_mem, total_mem;
          gpu::mem_get_info(&free_mem, &total_mem);

          // Note: pix_grid, D, V, W, cidx, cinfo already allocated
          size_t req_mem = 0;
          req_mem +=     Na  * Nh_ * Nw_ * sizeof(gpu::ComplexType<T>); // P
          req_mem +=     Nb  * Nh_ * Nw_ * sizeof(gpu::ComplexType<T>); // PW
          req_mem +=     Ne  * Nh_ * Nw_ * sizeof(gpu::ComplexType<T>); // E
          req_mem += 5 *             Nw_ * sizeof(gpu::ComplexType<T>); // d_ws, d_ps, d_pws, d_vs, d_es
          req_mem += 2 * Nl_ * Nh_ * Nw_ * sizeof(T);                   // stats_std_cum, stats_lsq_cum
          
          nsplit_ = (size_t) std::ceil(req_mem / (free_mem * 0.90));
          //nsplit_ = 5;
          printf("Info: GPU nsplit_ = %d\n", nsplit_); fflush(stdout);

          if (nsplit_ > 1) {
              nw_chunck_ = Nw_ / nsplit_;
              largest_chunck_ = Nw_ - (nsplit_ - 1) * nw_chunck_;
              req_mem /= Nw_;
              req_mem *= largest_chunck_;
              assert(req_mem < free_mem);
          }
          
          if (nsplit_ == 1) { // If nsplit_ == 1, only d2h after last iteration (triggered by client)
              stats_size_ =  Nl_ * Nh_ * Nw_;
              gpu::check_status(gpu::host_register(stats_std_cum_, stats_size_ * sizeof(T), 0x00));
              gpu::check_status(gpu::host_register(stats_lsq_cum_, stats_size_ * sizeof(T), 0x00));
              stats_std_cum_buff_ = create_buffer<T>(ctx_->allocators().gpu(), stats_size_);
              gpu::check_status(gpu::memset_async(stats_std_cum_buff_.get(), 0,
                                                  stats_size_ * sizeof(T), ctx_->gpu_stream()));
              stats_lsq_cum_buff_ = create_buffer<T>(ctx_->allocators().gpu(), stats_size_);
              gpu::check_status(gpu::memset_async(stats_lsq_cum_buff_.get(), 0,
                                                  stats_size_ * sizeof(T), ctx_->gpu_stream()));
          } else { // allocate a buffer of pinned memory based on largest chunck
              stats_size_ =  Nl_ * Nh_ * largest_chunck_;
              stats_std_buff = create_buffer<T>(ctx_->allocators().gpu(), stats_size_);
              stats_lsq_buff = create_buffer<T>(ctx_->allocators().gpu(), stats_size_);
              stats_std_pinned_buff = create_buffer<T>(ctx_->allocators().pinned(), stats_size_);
              stats_lsq_pinned_buff = create_buffer<T>(ctx_->allocators().pinned(), stats_size_);
          }
          
          first_epoch_ = false;
      }

      if (nsplit_ > 1) {

        for (int i=0; i<nsplit_; i++) {

          size_t Nws = i * nw_chunck_;
          size_t Nwe = i == nsplit_ - 1 ? Nw_ : Nws + nw_chunck_;

          // With nsplit_ > 1, reset stats as default behaviour is to accumulate
          gpu::check_status(gpu::memset_async(stats_std_buff.get(), 0,
                                              stats_size_ * sizeof(T), ctx_->gpu_stream()));
          gpu::check_status(gpu::memset_async(stats_lsq_buff.get(), 0,
                                              stats_size_ * sizeof(T), ctx_->gpu_stream()));

          standard_synthesizer_gpu<T>(*ctx_.get(), wl_, pix_grid_buff_.get() + 3 * Nh_ * Nws,
                                      xyz_buff.get(), d_buff.get(), v_buff.get(), w_buff.get(),
                                      cidx_buff.get(), cinfo_buff.get(), Na, Nb, Ne, Nh_, Nl_,
                                      Nws, Nwe, largest_chunck_,
                                      stats_std_buff.get(), stats_lsq_buff.get());
          
          // Copy partial epoch-wise stats from device to buffer on host
          size_t h2d_bytes = (Nwe - Nws) * Nl_ * Nh_ * sizeof(T);
          gpu::check_status(gpu::memcpy_async(stats_std_pinned_buff.get(), stats_std_buff.get(), h2d_bytes,
                                              gpu::flag::MemcpyDeviceToHost, ctx_->gpu_stream()));
          gpu::check_status(gpu::memcpy_async(stats_lsq_pinned_buff.get(), stats_lsq_buff.get(), h2d_bytes,
                                              gpu::flag::MemcpyDeviceToHost, ctx_->gpu_stream()));

          gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));

          // Accumulate partial stats to cum stats on host
#pragma omp parallel
          {
#pragma omp for
          for (size_t i=Nws; i<Nwe; i++) {
              for (size_t j=0; j<Nh_; j++) {
                  for (size_t k=0; k<Nl_; k++) {
                      stats_std_cum_[i * Nh_ * Nl_ + j * Nl_ + k] += stats_std_pinned_buff.get()[(i - Nws) * Nh_ * Nl_ + j * Nl_ + k];
                  }
              }
          }
#pragma omp for
          for (size_t i=Nws; i<Nwe; i++) {
              for (size_t j=0; j<Nh_; j++) {
                  for (size_t k=0; k<Nl_; k++) {
                      stats_lsq_cum_[i * Nh_ * Nl_ + j * Nl_ + k] += stats_lsq_pinned_buff.get()[(i - Nws) * Nh_ * Nl_ + j * Nl_ + k];
                  }
              }
          }
          }
        }

    } else {
        standard_synthesizer_gpu<T>(*ctx_.get(), wl_, pix_grid_buff_.get(),
                                    xyz_buff.get(), d_buff.get(), v_buff.get(), w_buff.get(),
                                    cidx_buff.get(), cinfo_buff.get(), Na, Nb, Ne, Nh_, Nl_,
                                    0, Nw_, Nw_, stats_std_cum_buff_.get(), stats_lsq_cum_buff_.get());

        // Copy cumulated stats back to host if requested
        if (d2h) {
            size_t N = Nl_ * Nh_ * Nw_;
            gpu::check_status(gpu::memcpy_async(stats_std_cum_, stats_std_cum_buff_.get(), N * sizeof(T),
                                                gpu::flag::MemcpyDeviceToHost, ctx_->gpu_stream()));
            gpu::check_status(gpu::memcpy_async(stats_lsq_cum_, stats_lsq_cum_buff_.get(), N * sizeof(T),
                                                gpu::flag::MemcpyDeviceToHost, ctx_->gpu_stream()));
            gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));
        }

      }

    gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));

#else
      throw GPUSupportError();
#endif
    }
  }

  T wl_;
  size_t Nl_, Nh_, Nw_, pix_grid_size_;
  const T* pix_grid_;
  T *stats_std_cum_, *stats_lsq_cum_;
  std::shared_ptr<ContextInternal> ctx_;
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
  size_t nsplit_ = 1, nw_chunck_, largest_chunck_, stats_size_;
  bool first_epoch_ = true;
  BufferType<T> pix_grid_buff_;
  BufferType<T> stats_std_buff, stats_lsq_buff;
  BufferType<T> stats_std_pinned_buff, stats_lsq_pinned_buff;
  BufferType<T> stats_std_cum_buff_, stats_lsq_cum_buff_; // if nsplit_ == 1
#endif
};

SS::SS(const Context &ctx, const double wl,
       const size_t Nl, const size_t Nh, const size_t Nw,
       const double* pix_grid, double* stats_std_cum, double* stats_lsq_cum)
    : ss_(new SSInternal<double>(InternalContextAccessor::get(ctx),
                                 wl, Nl, Nh, Nw, pix_grid,
                                 stats_std_cum, stats_lsq_cum),
          [](auto &&ptr) {
              delete reinterpret_cast<SSInternal<double> *>(ptr);
          }) {}
    
void SS::execute(const double* d, const std::complex<double>* v, const double* xyz,
                 const std::complex<double>* w, const std::size_t* c_idx,
                 const size_t Na, const size_t Ne, const size_t Nb, const bool d2h) {
    reinterpret_cast<SSInternal<double> *>(ss_.get())->exec(d, v, xyz, w, c_idx, Na, Ne, Nb, d2h);
}
    
SSf::SSf(const Context &ctx, const float wl,
         const size_t Nl, const size_t Nh, const size_t Nw,
         const float* pix_grid, float* stats_std_cum, float* stats_lsq_cum)
    : ss_(new SSInternal<float>(InternalContextAccessor::get(ctx),
                                wl, Nl, Nh, Nw, pix_grid,
                                stats_std_cum, stats_lsq_cum),
          [](auto &&ptr) {
              delete reinterpret_cast<SSInternal<float> *>(ptr);
          }) {}
    
void SSf::execute(const float* d, const std::complex<float>* v, const float* xyz,
                  const std::complex<float>* w, const std::size_t* c_idx,
                  const size_t Na, const size_t Ne, const size_t Nb, const bool d2h) {
    reinterpret_cast<SSInternal<float> *>(ss_.get())->exec(d, v, xyz, w, c_idx, Na, Ne, Nb, d2h);
}

} // namespace bluebild

