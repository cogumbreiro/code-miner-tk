CPP = clang++ -O3

all: sal2txt libsqlitefunctions.so

rapidjson:
	git clone https://github.com/Tencent/rapidjson/

sal2txt: src/sal2txt.cpp rapidjson
	$(CPP) -std=c++11 -I rapidjson/include -o sal2txt $<
	strip sal2txt

libsqlitefunctions.so:
	$(CC) -fPIC -lm -shared extension-functions.c -o libsqlitefunctions.so

clean:
	rm -f sal2txt
