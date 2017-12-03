CPP = clang++ -O3

all: sal2txt

rapidjson:
	git clone https://github.com/Tencent/rapidjson/

sal2txt: sal2txt.cpp rapidjson
	$(CPP) -I rapidjson/include -o sal2txt sal2txt.cpp
	strip sal2txt

clean:
	rm -f sal2txt
