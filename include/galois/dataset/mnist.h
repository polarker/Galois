#ifndef _GALOIS_MNIST_H_
#define _GALOIS_MNIST_H_

#include "galois/narray.h"
#include <zlib.h>
#include <string>

using namespace std;

namespace mnist
{
    
    class GzipFile
    {
    private:
        gzFile fp;
        
    public:
        GzipFile(const char *file, const char *mode) {
            fp = gzopen(file, mode);
            assert(fp);
        }
        ~GzipFile() {
            if (fp) {
                gzclose(fp);
            }
        }
        
        int32_t read_int() {
            uint8_t buf[4];
            assert(gzread(fp, buf, sizeof(buf)) == sizeof(buf));
            return int32_t(buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
        }
        
        uint8_t read_byte() {
            uint8_t b;
            assert(gzread(fp, &b, sizeof(b)) == sizeof(b));
            return b;
        }
    };
    
    template<typename T>
    gs::SP_NArray<T> read_images(const string &file_name, int num_samples=INT_MAX) {
        GzipFile gf(file_name.c_str(), "rb");
        int magic = gf.read_int();
        int count = gf.read_int();
        int rows  = gf.read_int();
        int cols  = gf.read_int();
        cout << "Image: magic=" << magic << ", count=" << count << ", rows=" << rows << ", cols=" << cols << endl;
        
        if (num_samples == INT_MAX) {
            num_samples = count;
        } else {
            assert(0 < num_samples && num_samples <= count);
        }
        
        auto res = make_shared<gs::NArray<T>>(num_samples, rows*cols);
        auto res_ptr = res->get_data();
        for (int i = 0; i < num_samples*rows*cols; i++) {
            res_ptr[i] = T(gf.read_byte()) / T(256);
        }
        
        return res;
    }
    
    template<typename T>
    gs::SP_NArray<T> read_labels(const string &file_name,  int num_samples=INT_MAX) {
        GzipFile gf(file_name.c_str(), "rb");
        int magic = gf.read_int();
        int count = gf.read_int();
        cout << "Lable: magic=" << magic << ", count=" << count << endl;
        
        if (num_samples == INT_MAX) {
            num_samples = count;
        } else {
            assert(0 < num_samples && num_samples <= count);
        }
        
        auto res = make_shared<gs::NArray<T>>(num_samples);
        auto res_ptr = res->get_data();
        for (int i = 0; i < num_samples; i++) {
            res_ptr[i] = T(gf.read_byte());
        }
        return res;
    }
    
}

#endif
