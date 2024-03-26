#include <mpi.h>
#include "mk5mode_gpu.cuh"
#include "gpumode_kernels.cuh"
#include "gpudecode.cuh"
//#include "mk5.h"
#include "alert.h"
#include <iostream>
#include <bitset>
#include <unistd.h>

#define NOT_SUPPORTED(x) { std::cerr << "Whoops, we don't support this on the GPU: " << x << std::endl; exit(1); }

Mk5_GPUMode::Mk5_GPUMode(Configuration * conf, int confindex, int dsindex, int recordedbandchan, int chanstoavg, int bpersend, int gsamples, int nrecordedfreqs, double recordedbw, double * recordedfreqclkoffs, double * recordedfreqclkoffsdelta, double * recordedfreqphaseoffs, double * recordedfreqlooffs, int nrecordedbands, int nzoombands, int nbits, Configuration::datasampling sampling, Configuration::complextype tcomplex, bool fbank, bool linear2circular, int fringerotorder, int arraystridelen, bool cacorrs, int framebytes, int framesamples, Configuration::dataformat format)
  : GPUMode(conf, confindex, dsindex, recordedbandchan, chanstoavg, bpersend, gsamples, nrecordedfreqs, recordedbw, recordedfreqclkoffs, recordedfreqclkoffsdelta, recordedfreqphaseoffs, recordedfreqlooffs, nrecordedbands, nzoombands, nbits, sampling, tcomplex, recordedbandchan*2+4, fbank, linear2circular, fringerotorder, arraystridelen, cacorrs, recordedbw*2)
{
  char formatname[64];

  fanout = config->genMk5FormatName(format, nrecordedbands, recordedbw, nbits, sampling, framebytes, conf->getDDecimationFactor(confindex, dsindex), config->getDAlignmentSeconds(confindex, dsindex), conf->getDNumMuxThreads(confindex, dsindex), formatname);
  invalid = 0;

  if(fanout < 0)
    initok = false;
  else
  {
    // since we allocated the max amount of space needed above, we need to change
    // this to the number actually needed.
    this->framesamples = framesamples;
    if (usecomplex) {
      unpacksamples = recordedbandchan;
      samplestounpack = recordedbandchan;
    } else {
      unpacksamples = recordedbandchan*2;
      samplestounpack = recordedbandchan*2;
    }
    //create the mark5_stream used for unpacking
    mark5stream = new_mark5_stream( new_mark5_stream_unpacker(0), new_mark5_format_generic_from_string(formatname) );
    if(mark5stream == 0)
    {
      cfatal << startl << "Mk5_GPUMode::Mk5_GPUMode : mark5stream is null" << endl;
      initok = false;
    }
    else
    {
      if(conf->isNetwork(dsindex))
        mark5stream->blanker = blanker_none;
      if(mark5stream->samplegranularity > 1)
        samplestounpack += mark5stream->samplegranularity;
      string orig_streamname(mark5stream->streamname);
      sprintf(mark5stream->streamname, "DS%d <%s>", dsindex, orig_streamname.c_str());
      if(framesamples != mark5stream->framesamples)
      {
        cfatal << startl << "Mk5_GPUMode::Mk5_GPUMode : framesamples inconsistent (told " << framesamples << "/ stream says " << mark5stream->framesamples << ") - for stream index " << dsindex << endl;
        initok = false;
      }
      else
      {
        this->framesamples = mark5stream->framesamples;
      }
      if(format == Configuration::INTERLACEDVDIF)
      {
        invalid = new int[nrecordedbands];
        perbandweights = new f32*[config->getNumBufferedFFTs(configindex)];
        for(int i=0;i<config->getNumBufferedFFTs(configindex);++i)
        {
          perbandweights[i] = new f32[nrecordedbands];
          for(int b = 0; b < nrecordedbands; ++b)
          {
            perbandweights[i][b] = 0.0;
          }
        }
      }
    }
  }
}

Mk5_GPUMode::~Mk5_GPUMode()
{
  delete_mark5_stream(mark5stream);
  if(invalid)
  {
    delete [] invalid;
  }
}

float Mk5_GPUMode::unpack(int sampleoffset, int subloopindex)
{
  float goodsamples = 0;
  int mungedoffset = 0;

  //work out where to start from
  unpackstartsamples = sampleoffset - (sampleoffset % mark5stream->samplegranularity);

  //unpack one frame plus one FFT size worth of samples
  if(usecomplex) 
  {
    NOT_SUPPORTED("unpack - usecomplex");
  }
  if(mark5stream->samplegranularity > 1)
    { // CHRIS not sure what this is mean to do
      // WALTER: unpacking of some mark5 modes (those with granularity > 1) must be unpacked not as individual samples but in groups of sample granularity
    int erasedsamples = 0;

    mungedoffset = sampleoffset % mark5stream->samplegranularity;
    for(int i = 0; i < mungedoffset; i++) {
      for(int b = subloopindex * numrecordedbands; b < subloopindex * numrecordedbands + mark5stream->nchan; ++b) {
        if(unpackedarrays_gpu->ptr()[b][i] != 0.0) {
            unpackedarrays_gpu->ptr()[b][i] = 0.0;
          erasedsamples++;
        }
      }
    }
    for(int i = unpacksamples + mungedoffset; i < samplestounpack; i++) {
      for(int b = subloopindex * numrecordedbands; b < subloopindex * numrecordedbands + mark5stream->nchan; ++b) {
        if(unpackedarrays_gpu->ptr()[b][i] != 0.0) {
            unpackedarrays_gpu->ptr()[b][i] = 0.0;
          erasedsamples++;
        }
      }
    }
    goodsamples -= erasedsamples/(float)(mark5stream->nchan);
  }
  if(perbandweights)
  {
      if(usecomplex)
      {
          NOT_SUPPORTED("unpack - usecomplex");
      }
      else
      {
          blank_vdif_EDV4(data, unpackstartsamples, &unpackedarrays_gpu->ptr()[subloopindex * numrecordedbands], samplestounpack, invalid);
      }

      int totalinvalid = 0;
      for(int b = 0; b < mark5stream->nchan; ++b)
      {
          perbandweights[subloopindex][b] = (goodsamples - invalid[b])/(float)unpacksamples;
          totalinvalid += invalid[b];
      }

      goodsamples -= (float)totalinvalid/(float)(mark5stream->nchan);
  }

  if(goodsamples < 0)
  {
    cerror << startl << "Error trying to unpack Mark5 format data at sampleoffset " << sampleoffset << " from data seconds " << datasec << " plus " << datans << " ns!!!" << endl;
    goodsamples = 0;
    for(int b = 0; b < mark5stream->nchan; ++b)
      invalid[b] = 0;
  }

  return goodsamples/(float)unpacksamples;
}

// This unpack_all call (like all GPUMode unpack_alls) unpacks everything
// (except the header of course), including the unnecessary period between the
// start of the frame and the first sample in the observation. (We do this to
// avoid a corner-case where unpacking has to begin inside a word, or worse,
// inside a byte)
void Mk5_GPUMode::unpack_all(int framestounpack, int frame_size) {
    // Hacky little workaround to get the stream struct back !! May not be needed !!
  /*
    mark5_stream *tmp_mk5stream;
    cudaMallocManaged(&tmp_mk5stream, sizeof(mark5_stream));
    *tmp_mk5stream = *mark5stream;
    */

    std::cout << "frames to unpack: " << framestounpack << std::endl;

    const int BYTES_PER_WORD = 4;
    const int unpack_threads = 256;
    const int total_bytes = framestounpack * frame_size;
    assert(total_bytes % BYTES_PER_WORD == 0); // VDIF must always be an integer number of words
    const int total_words = total_bytes / BYTES_PER_WORD;
    const int blocks = (total_bytes + unpack_threads - 1) / unpack_threads;

    const int HEADERLENGTH_BYTES = 4 * 8;
    const int payloadlength_bytes = frame_size - HEADERLENGTH_BYTES;
    const int payloadlength_words = payloadlength_bytes / BYTES_PER_WORD;

    std::cout << " cuda blocks: " << blocks << " ; total_bytes: " << total_bytes << "; total_words: " << total_words << std::endl;
    gpu_unpack<<<blocks, unpack_threads, 0, cuStream>>>(
        (char*)packeddata_gpu->gpuPtr(),
        unpackedarrays_gpu->gpuPtr(),
        valid_frames->gpuPtr(),
        payloadlength_words,
        frame_size,
        total_words
      );

    /*
    int unpack_blocks = (framestounpack + unpack_threads - 1) / unpack_threads;

    cudaStreamSynchronize(cuStream);
    std::cout << "call gpu_unpack - unpack_blocks = " << unpack_blocks << "; unpack_threads = " << unpack_threads << std::endl;
    gpu_unpack<<<unpack_blocks, unpack_threads, 0, cuStream>>>(tmp_mk5stream, packeddata_gpu->gpuPtr(), unpackedarrays_gpu->gpuPtr(), framestounpack, valid_frames->gpuPtr());
    */

    /*
    cudaMemset(valid_frames->gpuPtr(), 1, framestounpack);

    // Unfortunately we have to block here since we need the valid frames to find the correct dataweights
    valid_frames->sync();
    valid_frames->copyToHost();
    valid_frames->sync();
    */

    //cudaFree(tmp_mk5stream);

}
// vim: shiftwidth=2:softtabstop=2:expandtab
