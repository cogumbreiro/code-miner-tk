CPP = clang++ -O3

all: sal2txt

rapidjson:
	git clone https://github.com/Tencent/rapidjson/

sal2txt: src/sal2txt.cpp rapidjson
	$(CPP) -std=c++11 -I rapidjson/include -o sal2txt $<
	strip sal2txt

clean:
	rm -f sal2txt
