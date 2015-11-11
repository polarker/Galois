#ifndef _GALOIS_CHARNN_READER_H_
#define _GALOIS_CHARNN_READER_H_

#include "galois/narray.h"
#include "galois/utils.h"
#include <fstream>
#include <string>
#include <map>

using namespace std;

namespace chartxt
{
    
    template<typename T>
    class Article
    {
    private:
        int num_diff_chars = 0;
        map<char, int> char2int = {};
        map<int, char> int2char = {};
        
        int sequence_length = 0;
        gs::SP_NArray<T> int_sequence = nullptr;
        gs::SP_NArray<T> vectorized_sequence = nullptr;
        gs::SP_NArray<T> target_sequence = nullptr;
        
    public:
        explicit Article(const string &file_name) {
            ifstream fin;
            char ch;
            int num_chars;
            
            fin.open(file_name);
            CHECK(fin.is_open(), "can not open file");
            num_chars = 0;
            while (fin >> noskipws >> ch) {
                num_chars += 1;
                if (char2int.count(ch) == 0) {
                    int idx = char2int.size();
                    char2int[ch] = idx;
                    int2char[idx] = ch;
                }
            }
            fin.close();
            
            num_diff_chars = char2int.size();
            sequence_length = num_chars - 1;
            int_sequence = make_shared<gs::NArray<T>>(num_chars);
            vectorized_sequence = make_shared<gs::NArray<T>>(num_chars-1, num_diff_chars);
            target_sequence = make_shared<gs::NArray<T>>(num_chars-1);
            
            cout << "size of chars: " << num_chars << endl;
            cout << "size of different chars: " << char2int.size() << endl;
            fin.open(file_name);
            CHECK(fin.is_open(), "can not open file");
            auto int_sequence_ptr = int_sequence->get_data();
            auto vectorized_sequence_ptr = vectorized_sequence->get_data();
            auto target_sequence_ptr = target_sequence->get_data();
            for (int i = 0; i < num_chars; i++) {
                fin >> noskipws >> ch;
                int idx = char2int[ch];
                
                int_sequence_ptr[i] = idx;
                if (i < num_chars - 1) {
                    for (int j = 0; j < num_diff_chars; j++) {
                        if (j == idx) {
                            vectorized_sequence_ptr[i*num_diff_chars + j] = 1;
                        } else {
                            vectorized_sequence_ptr[i*num_diff_chars + j] = 0;
                        }
                    }
                }
                if (i > 0) {
                    target_sequence_ptr[i-1] = idx;
                }
            }
            fin.close();
        }
        Article() = delete;
        Article(const Article&) = delete;
        Article& operator=(Article &) = delete;
        
        int get_num_diff_chars() {
            return num_diff_chars;
        }
        
//        int get_sequence_length {
//            return sequence_length;
//        }
        
        gs::SP_NArray<T> get_vectorized_sequence() {
            return vectorized_sequence;
        }
        
        gs::SP_NArray<T> get_target_sequence() {
            return target_sequence;
        }
    };
    
}

#endif
