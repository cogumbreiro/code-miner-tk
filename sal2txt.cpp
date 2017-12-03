#include "rapidjson/reader.h"
#include "rapidjson/istreamwrapper.h"
#include <iostream>
using namespace rapidjson;
using namespace std;
struct Sal2Txt : public BaseReaderHandler<UTF8<>, Sal2Txt> {
    bool in_call;
    bool in_seq;
    uint arr_depth;
    uint obj_depth;
    Sal2Txt() : in_call(false), in_seq(false), arr_depth(0), obj_depth(0) {}
    bool Null() { return true; }
    bool Bool(bool b) { return true; }
    bool Int(int i) { return true; }
    bool Uint(unsigned u) { return true; }
    bool Int64(int64_t i) { return true; }
    bool Uint64(uint64_t u) { return true; }
    bool Double(double d) { return true; }
    bool String(const char* str, SizeType length, bool copy) {
        if (in_call) {
            cout << str << " ";
            in_call = false;
        }
        return true;
    }
    bool Key(const char* str, SizeType length, bool copy) {
        if (! in_seq) {
            arr_depth = 0;
            obj_depth = 0;
            in_seq = strcmp(str, "sequence") == 0;
        } else {
            in_call = strcmp(str, "call") == 0;
        }
        return true;
    }
    bool StartArray() { if (in_seq) { arr_depth++; } return true; }
    bool EndArray(SizeType elementCount) {
        if (in_seq) {
            arr_depth--;
            if (arr_depth == 0) {
                std::cout << "$END" << std::endl;
                in_seq = false;
            }
        }
        return true;
    }
    bool StartObject() { if (in_seq) obj_depth++; return true; }
    bool EndObject(SizeType memberCount) { if (in_seq) obj_depth--; return true; }
};

int main() {
    Sal2Txt handler;
    Reader reader;
    IStreamWrapper isw(std::cin);
    reader.Parse(isw, handler);
}
