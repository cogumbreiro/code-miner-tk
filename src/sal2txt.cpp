#include "rapidjson/reader.h"
#include "rapidjson/istreamwrapper.h"
#include "args.hxx"
#include <iostream>
#include <set>
using namespace rapidjson;
//using namespace std;
struct Sal2Txt : public BaseReaderHandler<UTF8<>, Sal2Txt> {
    std::string eol;
    bool inc_prefix;
    std::set<std::string> stop_words;
    bool in_call;
    bool in_seq;
    int count;
    uint arr_depth;
    uint obj_depth;
    Sal2Txt(std::string eol, bool inc_prefix, std::set<std::string> stop_words) :
        eol(eol),
        inc_prefix(inc_prefix),
        stop_words(stop_words),
        in_call(false),
        in_seq(false),
        count(0),
        arr_depth(0),
        obj_depth(0)
        {}
    bool Null() { return true; }
    bool Bool(bool b) { return true; }
    bool Int(int i) { return true; }
    bool Uint(unsigned u) { return true; }
    bool Int64(int64_t i) { return true; }
    bool Uint64(uint64_t u) { return true; }
    bool Double(double d) { return true; }
    bool String(const char* str, SizeType length, bool copy) {
        if (in_call) {
            // ignore calls that start with '_'
            std::string key = str;
            if (key.length() > 0 && (inc_prefix || str[0] != '_') && stop_words.count(key) == 0) {
                count++;
                std::cout << str << " ";
            }
            in_call = false;
        }
        return true;
    }
    bool Key(const char* str, SizeType length, bool copy) {
        if (! in_seq) {
            arr_depth = 0;
            obj_depth = 0;
            count = 0;
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
                if (count) {
                    std::cout << eol << std::endl;
                }
                in_seq = false;
            }
        }
        return true;
    }
    bool StartObject() { if (in_seq) obj_depth++; return true; }
    bool EndObject(SizeType memberCount) { if (in_seq) obj_depth--; return true; }
};

int main(int argc, const char **argv) {
    args::ArgumentParser parser("Converts a Salento JSON file into a plain text file.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> eol(parser, "end-of-line", "The end-of-line word. Default: $END", {"eol"}, "$END");
    args::PositionalList<std::string> sw(parser, "stop-word", "Stop words to be ignored.", {"stop-words"});
    args::Flag inc_prefix(parser, "include-prefix", "Show terms starting with prefix '_'. By default functions prefixed by '_' are ignored.", {"include-prefix"});
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    std::set<std::string> stop_words;
    for (auto &&word : sw) {
        stop_words.insert(word);
    }

    Sal2Txt handler(args::get(eol), inc_prefix, stop_words);
    Reader reader;
    IStreamWrapper isw(std::cin);
    reader.Parse(isw, handler);
}
