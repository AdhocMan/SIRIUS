
template <int direction>
void FFT3D::transform_z_parallel(std::vector<int> const& sendcounts, std::vector<int> const& sdispls,
                                 std::vector<int> const& recvcounts, std::vector<int> const& rdispls,
                                 int num_z_cols_local)
{
    if (comm_.size() > 1)
    {
        Timer t1("fft|a2a_internal");
        comm_.alltoall(fftw_buffer_, &sendcounts[0], &sdispls[0], &fft_buffer_aux_(0), &recvcounts[0], &rdispls[0]);
        t1.stop();
    
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int i = 0; i < num_z_cols_local; i++)
            {
                for (int rank = 0; rank < comm_.size(); rank++)
                {
                    int lsz = (int)spl_z_.local_size(rank);
    
                    memcpy(&fftw_buffer_z_[tid][spl_z_.global_offset(rank)], 
                           &fft_buffer_aux_(spl_z_.global_offset(rank) * num_z_cols_local + i * lsz),
                           lsz * sizeof(double_complex));
                }
                switch (direction)
                {
                    case 1:
                    {
                        fftw_execute(plan_backward_z_[tid]);
                        break;
                    }
                    case -1:
                    {
                        fftw_execute(plan_forward_z_[tid]);
                        break;
                    }
                    default:
                    {
                        TERMINATE("wrong direction");
                    }
                }
                for (int rank = 0; rank < comm_.size(); rank++)
                {
                    int lsz = (int)spl_z_.local_size(rank);
    
                    memcpy(&fft_buffer_aux_(spl_z_.global_offset(rank) * num_z_cols_local + i * lsz),
                           &fftw_buffer_z_[tid][spl_z_.global_offset(rank)], 
                           lsz * sizeof(double_complex));
                }
            }
        }
    
        t1.start();
        comm_.alltoall(&fft_buffer_aux_(0), &recvcounts[0], &rdispls[0], fftw_buffer_, &sendcounts[0], &sdispls[0]);
    }
}

template <int direction>
void FFT3D::transform_xy_parallel(std::vector< std::pair<int, int> > const& z_sticks_coord__)
{
    int size_xy = fft_grid_.size(0) * fft_grid_.size(1);
    if (pu_ == CPU)
    {
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int iz = 0; iz < local_size_z_; iz++)
            {
                switch (direction)
                {
                    case 1:
                    {
                        memset(fftw_buffer_xy_[tid], 0, sizeof(double_complex) * size_xy);
                        for (int n = 0; n < (int)z_sticks_coord__.size(); n++)
                        {
                            int x = z_sticks_coord__[n].first;
                            int y = z_sticks_coord__[n].second;

                            fftw_buffer_xy_[tid][x + y * fft_grid_.size(0)] = fft_buffer_aux_(iz + local_size_z_ * n);
                        }
                        fftw_execute(plan_backward_xy_[tid]);
                        memcpy(&fftw_buffer_[iz * size_xy], fftw_buffer_xy_[tid], sizeof(fftw_complex) * size_xy);
                        break;
                    }
                    case -1:
                    {
                        memcpy(fftw_buffer_xy_[tid], &fftw_buffer_[iz * size_xy], sizeof(fftw_complex) * size_xy);
                        fftw_execute(plan_forward_xy_[tid]);
                        for (int n = 0; n < (int)z_sticks_coord__.size(); n++)
                        {
                            int x = z_sticks_coord__[n].first;
                            int y = z_sticks_coord__[n].second;
                            fft_buffer_aux_(iz + local_size_z_ * n) = fftw_buffer_xy_[tid][x + y * fft_grid_.size(0)];
                        }
                        break;
                    }
                    default:
                    {
                        TERMINATE("wrong direction");
                    }
                }
            }
        }
    }
    #ifdef __GPU
    if (pu_ == GPU)
    {
        if (direction == 1)
        {
            memset(fftw_buffer_, 0, sizeof(double_complex) * local_size());
            #pragma omp parallel for num_threads(num_fft_workers_)
            for (int iz = 0; iz < local_size_z_; iz++)
            {
                for (int n = 0; n < (int)z_sticks_coord__.size(); n++)
                {
                    int x = z_sticks_coord__[n].first;
                    int y = z_sticks_coord__[n].second;
                    fftw_buffer_[x + y * size(0) + iz * size_xy] = fft_buffer_aux_(iz + local_size_z_ * n);
                }
            }
            cufft_buf_.copy_to_device();
            cufft_backward_transform(cufft_plan_xy_, cufft_buf_.at<GPU>());
            cufft_buf_.copy_to_host();
        }
        if (direction == -1)
        {
            cufft_buf_.copy_to_device();
            cufft_forward_transform(cufft_plan_xy_, cufft_buf_.at<GPU>());
            cufft_buf_.copy_to_host();
            #pragma omp parallel for num_threads(num_fft_workers_)
            for (int iz = 0; iz < local_size_z_; iz++)
            {
                for (int n = 0; n < (int)z_sticks_coord__.size(); n++)
                {
                    int x = z_sticks_coord__[n].first;
                    int y = z_sticks_coord__[n].second;
                    fft_buffer_aux_(iz + local_size_z_ * n) = fftw_buffer_[x + y * size(0) + iz * size_xy];
                }
            }
        }
    }
    #endif
}

template <int direction>
void FFT3D::transform_z_serial(std::vector< std::pair<int, int> > const& z_sticks_coord__)
{
    int size_xy = fft_grid_.size(0) * fft_grid_.size(1);
    if (pu_ == CPU)
    {
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int i = 0; i < (int)z_sticks_coord__.size(); i++)
            {
                int x = z_sticks_coord__[i].first;
                int y = z_sticks_coord__[i].second;
    
                for (int z = 0; z < fft_grid_.size(2); z++)
                {
                    fftw_buffer_z_[tid][z] = fftw_buffer_[x + y * fft_grid_.size(0) + z * size_xy];
                }
                switch (direction)
                {
                    case 1:
                    {
                        fftw_execute(plan_backward_z_[tid]);
                        break;
                    }
                    case -1:
                    {
                        fftw_execute(plan_forward_z_[tid]);
                        break;
                    }
                    default:
                    {
                        TERMINATE("wrong direction");
                    }
                }
                for (int z = 0; z < fft_grid_.size(2); z++)
                {
                    fftw_buffer_[x + y * fft_grid_.size(0) + z * size_xy] = fftw_buffer_z_[tid][z];
                }
            }
        }
    }
    #ifdef __GPU
    if (pu_ == GPU)
    {
        STOP();
        //for (int j = 0; j < (int)z_sticks_coord__.size(); j++)
        //{
        //    int x = z_sticks_coord__[j].first;
        //    int y = z_sticks_coord__[j].second;
        //    int stream_id = j % get_num_cuda_streams();
        //    switch (direction)
        //    {
        //        case 1:
        //        {
        //            cufft_backward_transform(cufft_plan_z_[stream_id], cufft_buf_.at<GPU>(x + y * size(0)));
        //            break;
        //        }
        //        case -1:
        //        {
        //            cufft_forward_transform(cufft_plan_z_[stream_id], cufft_buf_.at<GPU>(x + y * size(0)));
        //            break;
        //        }
        //        default:
        //        {
        //            TERMINATE("wrong direction");
        //        }
        //   }
        //}
        //for (int i = 0; i < get_num_cuda_streams(); i++)
        //    cuda_stream_synchronize(i);
    }
    #endif
}

template <int direction>
void FFT3D::transform_xy_serial()
{
    int size_xy = fft_grid_.size(0) * fft_grid_.size(1);
    if (pu_ == CPU)
    {
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int z = 0; z < fft_grid_.size(2); z++)
            {
                memcpy(fftw_buffer_xy_[tid], &fftw_buffer_[z * size_xy], sizeof(double_complex) * size_xy);
                switch (direction)
                {
                    case 1:
                    {
                        fftw_execute(plan_backward_xy_[tid]);
                        break;
                    }
                    case -1:
                    {
                        fftw_execute(plan_forward_xy_[tid]);
                        break;
                    }
                    default:
                    {
                        TERMINATE("wrong direction");
                    }
                }
                memcpy(&fftw_buffer_[z * size_xy], fftw_buffer_xy_[tid], sizeof(double_complex) * size_xy);
            }
        }
    }
    #ifdef __GPU
    if (pu_ == GPU)
    {
        STOP();
        //for (int z = 0; z < size(2); z++)
        //{
        //    int stream_id = z % get_num_cuda_streams();
        //    switch (direction)
        //    {
        //        case 1:
        //        {
        //            cufft_backward_transform(cufft_plan_xy_[stream_id], cufft_buf_.at<GPU>(z * size_xy));
        //            break;
        //        }
        //        case -1:
        //        {
        //            cufft_forward_transform(cufft_plan_xy_[stream_id], cufft_buf_.at<GPU>(z * size_xy));
        //            break;
        //        }
        //        default:
        //        {
        //            TERMINATE("wrong direction");
        //        }
        //    }
        //}
        //for (int i = 0; i < get_num_cuda_streams(); i++)
        //    cuda_stream_synchronize(i);
    }
    #endif
}

template <int direction>
void FFT3D::transform_z_serial(std::vector<z_column_descriptor> const& z_cols__, double_complex* data__)
{
    int size_xy = fft_grid_.size(0) * fft_grid_.size(1);
    if (pu_ == CPU)
    {
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(dynamic, 1)
            for (size_t i = 0; i < z_cols__.size(); i++)
            {
                int x = z_cols__[i].x;
                int y = z_cols__[i].y;
                int offset = z_cols__[i].offset;

                switch (direction)
                {
                    case 1:
                    {
                        //std::fill(fftw_buffer_z_[tid], fftw_buffer_z_[tid] + size(2), double_complex(0, 0));
                        std::memset(fftw_buffer_z_[tid], 0, fft_grid_.size(2) * sizeof(double_complex));
                        for (size_t j = 0; j < z_cols__[i].z.size(); j++)
                        {
                            fftw_buffer_z_[tid][z_cols__[i].z[j]] = data__[offset + j];
                        }
                        fftw_execute(plan_backward_z_[tid]);
                        for (int z = 0; z < fft_grid_.size(2); z++)
                        {
                            fftw_buffer_[x + y * fft_grid_.size(0) + z * size_xy] = fftw_buffer_z_[tid][z];
                        }
                        break;

                    }
                    case -1:
                    {
                        STOP();
                    }
                    default:
                    {
                        TERMINATE("wrong direction");
                    }
                }

    
                //for (int z = 0; z < size(2); z++)
                //{
                //    fftw_buffer_z_[tid][z] = fftw_buffer_[x + y * size(0) + z * size_xy];
                //}
                //switch (direction)
                //{
                //    case 1:
                //    {
                //        fftw_execute(plan_backward_z_[tid]);
                //        break;
                //    }
                //    case -1:
                //    {
                //        fftw_execute(plan_forward_z_[tid]);
                //        break;
                //    }
                //    default:
                //    {
                //        TERMINATE("wrong direction");
                //    }
                //}
                //for (int z = 0; z < size(2); z++)
                //{
                //    fftw_buffer_[x + y * size(0) + z * size_xy] = fftw_buffer_z_[tid][z];
                //}
            }
        }
    }
    #ifdef __GPU
    if (pu_ == GPU)
    {
        STOP();
        //for (int j = 0; j < (int)z_sticks_coord__.size(); j++)
        //{
        //    int x = z_sticks_coord__[j].first;
        //    int y = z_sticks_coord__[j].second;
        //    int stream_id = j % get_num_cuda_streams();
        //    switch (direction)
        //    {
        //        case 1:
        //        {
        //            cufft_backward_transform(cufft_plan_z_[stream_id], cufft_buf_.at<GPU>(x + y * size(0)));
        //            break;
        //        }
        //        case -1:
        //        {
        //            cufft_forward_transform(cufft_plan_z_[stream_id], cufft_buf_.at<GPU>(x + y * size(0)));
        //            break;
        //        }
        //        default:
        //        {
        //            TERMINATE("wrong direction");
        //        }
        //   }
        //}
        //for (int i = 0; i < get_num_cuda_streams(); i++)
        //    cuda_stream_synchronize(i);
    }
    #endif
}

template <int direction>
void FFT3D::transform_z_parallel(block_data_descriptor const& zcol_distr__,
                                 std::vector<z_column_descriptor> const& z_cols__,
                                 double_complex* data__)
{
    int rank = comm_.rank();
    int num_zcol_local = zcol_distr__.counts[rank];

    Timer t0("fft|z_parallel_local");
    #pragma omp parallel num_threads(num_fft_workers_)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < num_zcol_local; i++)
        {
            int icol = zcol_distr__.offsets[rank] + i;
            int offset = z_cols__[icol].offset;

            switch (direction)
            {
                case 1:
                {
                    /* clear z buffer */
                    //std::fill(fftw_buffer_z_[tid], fftw_buffer_z_[tid] + size(2), double_complex(0, 0));
                    std::memset(fftw_buffer_z_[tid], 0, fft_grid_.size(2) * sizeof(double_complex));
                    /* load z column into buffer */
                    for (size_t j = 0; j < z_cols__[icol].z.size(); j++)
                    {
                        fftw_buffer_z_[tid][z_cols__[icol].z[j]] = data__[offset + j];
                    }
                    /* perform local FFT transform of a column */
                    fftw_execute(plan_backward_z_[tid]);
                    /* redistribute z-column for a forthcoming all-to-all */ 
                    for (int r = 0; r < comm_.size(); r++)
                    {
                        int lsz = (int)spl_z_.local_size(r);
                    
                        std::memcpy(&fft_buffer_aux_(spl_z_.global_offset(r) * num_zcol_local + i * lsz),
                                    &fftw_buffer_z_[tid][spl_z_.global_offset(r)], 
                                    lsz * sizeof(double_complex));
                    }
                    break;

                }
                case -1:
                {
                    STOP();
                }
                default:
                {
                    TERMINATE("wrong direction");
                }
            }
        }
    }
    t0.stop();
    
    switch (direction)
    {
        case 1:
        {
            block_data_descriptor send(comm_.size());
            block_data_descriptor recv(comm_.size());
            for (int r = 0; r < comm_.size(); r++)
            {
                send.counts[r] = static_cast<int>(spl_z_.local_size(r) * zcol_distr__.counts[rank]);
                recv.counts[r] = static_cast<int>(spl_z_.local_size(rank) * zcol_distr__.counts[r]);
            }
            send.calc_offsets();
            recv.calc_offsets();
            
            Timer t1("fft|a2a_internal");
            comm_.alltoall(&fft_buffer_aux_(0), &send.counts[0], &send.offsets[0], fftw_buffer_, &recv.counts[0], &recv.offsets[0]);
            t1.stop();
            std::memcpy(&fft_buffer_aux_(0), fftw_buffer_, z_cols__.size() * local_size_z_ * sizeof(double_complex)); 
        }
    }
}

template <int direction>
void FFT3D::transform_xy_parallel(std::vector<z_column_descriptor> const& z_cols__)
{
    Timer t0("fft|xy_parallel");
    int size_xy = fft_grid_.size(0) * fft_grid_.size(1);
    if (pu_ == CPU)
    {
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int iz = 0; iz < local_size_z_; iz++)
            {
                switch (direction)
                {
                    case 1:
                    {
                        /* clear xy-buffer */
                        std::memset(fftw_buffer_xy_[tid], 0, sizeof(double_complex) * size_xy);
                        /* load z-columns into proper location */
                        for (size_t i = 0; i < z_cols__.size(); i++)
                        {
                            int x = z_cols__[i].x;
                            int y = z_cols__[i].y;

                            fftw_buffer_xy_[tid][x + y * fft_grid_.size(0)] = fft_buffer_aux_(iz + local_size_z_ * i);
                        }
                        /* execute local FFT transform */
                        fftw_execute(plan_backward_xy_[tid]);
                        /* copy xy plane to the fft buffer */
                        std::memcpy(&fftw_buffer_[iz * size_xy], fftw_buffer_xy_[tid], sizeof(fftw_complex) * size_xy);
                        break;
                    }
                    //case -1:
                    //{
                    //    memcpy(fftw_buffer_xy_[tid], &fftw_buffer_[iz * size_xy], sizeof(fftw_complex) * size_xy);
                    //    fftw_execute(plan_forward_xy_[tid]);
                    //    for (int n = 0; n < (int)z_sticks_coord__.size(); n++)
                    //    {
                    //        int x = z_sticks_coord__[n].first;
                    //        int y = z_sticks_coord__[n].second;
                    //        fft_buffer_aux_(iz + local_size_z_ * n) = fftw_buffer_xy_[tid][x + y * fft_grid_.size(0)];
                    //    }
                    //    break;
                    //}
                    default:
                    {
                        TERMINATE("wrong direction");
                    }
                }
            }
        }
    }
    //#ifdef __GPU
    //if (pu_ == GPU)
    //{
    //    if (direction == 1)
    //    {
    //        memset(fftw_buffer_, 0, sizeof(double_complex) * local_size());
    //        #pragma omp parallel for num_threads(num_fft_workers_)
    //        for (int iz = 0; iz < local_size_z_; iz++)
    //        {
    //            for (int n = 0; n < (int)z_sticks_coord__.size(); n++)
    //            {
    //                int x = z_sticks_coord__[n].first;
    //                int y = z_sticks_coord__[n].second;
    //                fftw_buffer_[x + y * size(0) + iz * size_xy] = fft_buffer_aux_(iz + local_size_z_ * n);
    //            }
    //        }
    //        cufft_buf_.copy_to_device();
    //        cufft_backward_transform(cufft_plan_xy_, cufft_buf_.at<GPU>());
    //        cufft_buf_.copy_to_host();
    //    }
    //    if (direction == -1)
    //    {
    //        cufft_buf_.copy_to_device();
    //        cufft_forward_transform(cufft_plan_xy_, cufft_buf_.at<GPU>());
    //        cufft_buf_.copy_to_host();
    //        #pragma omp parallel for num_threads(num_fft_workers_)
    //        for (int iz = 0; iz < local_size_z_; iz++)
    //        {
    //            for (int n = 0; n < (int)z_sticks_coord__.size(); n++)
    //            {
    //                int x = z_sticks_coord__[n].first;
    //                int y = z_sticks_coord__[n].second;
    //                fft_buffer_aux_(iz + local_size_z_ * n) = fftw_buffer_[x + y * size(0) + iz * size_xy];
    //            }
    //        }
    //    }
    //}
    //#endif
}



template <int direction>
void FFT3D::transform(Gvec const& gvec__, double_complex* data__)
{
    Timer t0("fft|transform");
    if (comm_.size() == 1)
    {
        switch (direction)
        {
            case 1:
            {
                std::memset(fftw_buffer_, 0, size() * sizeof(double_complex));
                transform_z_serial<1>(gvec__.z_columns(), data__);
                transform_xy_serial<1>();
                break;
            }
            default:
            {
                TERMINATE("wrong direction");
            }
        }   
    }
    else
    {
        switch (direction)
        {
            case 1:
            {
                transform_z_parallel<1>(gvec__.zcol_fft_distr(), gvec__.z_columns(), data__);
                transform_xy_parallel<1>(gvec__.z_columns());
                break;
            }
            default:
            {
                TERMINATE("wrong direction");
            }
        }   
    }
}

        
