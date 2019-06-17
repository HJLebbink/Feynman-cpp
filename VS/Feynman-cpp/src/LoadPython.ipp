//#pragma once
//
//#include <string>
//#include <stdexcept>
//#include <sstream>
//#include <vector>
//#include <cstdio>
//#include <typeinfo>
//#include <iostream>
//#include <cassert>
////#include <zlib.h>
//#include <map>
//#include <complex>
//#include <cstdlib>
//#include <algorithm>
//#include <cstring>
//#include <iomanip>
//
//namespace cnpy {
//	// see https://github.com/rogersce/cnpy
//
//	struct NpyArray {
//		char* data;
//		std::vector<unsigned int> shape;
//		unsigned int word_size;
//		bool fortran_order;
//		void destruct() { delete[] data; }
//	};
//
//	struct npz_t : public std::map<std::string, NpyArray>
//	{
//		void destruct()
//		{
//			npz_t::iterator it = this->begin();
//			for (; it != this->end(); ++it) (*it).second.destruct();
//		}
//	};
//
//	template<typename T> std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
//		//write in little endian
//		for (char byte = 0; byte < sizeof(T); byte++) {
//			char val = *((char*)&rhs + byte);
//			lhs.push_back(val);
//		}
//		return lhs;
//	}
//
//	template<typename T> std::string tostring(T i, int pad = 0, char padval = ' ') {
//		std::stringstream s;
//		s << i;
//		return s.str();
//	}
//
//	template<typename T> void npy_save(std::string fname, const T* data, const unsigned int* shape, const unsigned int ndims, std::string mode = "w") {
//		FILE* fp = NULL;
//
//		if (mode == "a") fp = fopen(fname.c_str(), "r+b");
//
//		if (fp) {
//			//file exists. we need to append to it. read the header, modify the array size
//			unsigned int word_size, tmp_dims;
//			unsigned int* tmp_shape = 0;
//			bool fortran_order;
//			parse_npy_header(fp, word_size, tmp_shape, tmp_dims, fortran_order);
//			assert(!fortran_order);
//
//			if (word_size != sizeof(T)) {
//				std::cout << "libnpy error: " << fname << " has word size " << word_size << " but npy_save appending data sized " << sizeof(T) << "\n";
//				assert(word_size == sizeof(T));
//			}
//			if (tmp_dims != ndims) {
//				std::cout << "libnpy error: npy_save attempting to append misdimensioned data to " << fname << "\n";
//				assert(tmp_dims == ndims);
//			}
//
//			for (int i = 1; i < ndims; i++) {
//				if (shape[i] != tmp_shape[i]) {
//					std::cout << "libnpy error: npy_save attempting to append misshaped data to " << fname << "\n";
//					assert(shape[i] == tmp_shape[i]);
//				}
//			}
//			tmp_shape[0] += shape[0];
//
//			fseek(fp, 0, SEEK_SET);
//			std::vector<char> header = create_npy_header(data, tmp_shape, ndims);
//			fwrite(&header[0], sizeof(char), header.size(), fp);
//			fseek(fp, 0, SEEK_END);
//
//			delete[] tmp_shape;
//		}
//		else {
//			fp = fopen(fname.c_str(), "wb");
//			std::vector<char> header = create_npy_header(data, shape, ndims);
//			fwrite(&header[0], sizeof(char), header.size(), fp);
//		}
//
//		unsigned int nels = 1;
//		for (int i = 0;i < ndims;i++) nels *= shape[i];
//
//		fwrite(data, sizeof(T), nels, fp);
//		fclose(fp);
//	}
//
//	template<typename T> void npz_save(std::string zipname, std::string fname, const T* data, const unsigned int* shape, const unsigned int ndims, std::string mode = "w")
//	{
//		//first, append a .npy to the fname
//		fname += ".npy";
//
//		//now, on with the show
//		FILE* fp = NULL;
//		unsigned short nrecs = 0;
//		unsigned int global_header_offset = 0;
//		std::vector<char> global_header;
//
//		if (mode == "a") fp = fopen(zipname.c_str(), "r+b");
//
//		if (fp) {
//			//zip file exists. we need to add a new npy file to it.
//			//first read the footer. this gives us the offset and size of the global header
//			//then read and store the global header. 
//			//below, we will write the the new data at the start of the global header then append the global header and footer below it
//			unsigned int global_header_size;
//			parse_zip_footer(fp, nrecs, global_header_size, global_header_offset);
//			fseek(fp, global_header_offset, SEEK_SET);
//			global_header.resize(global_header_size);
//			size_t res = fread(&global_header[0], sizeof(char), global_header_size, fp);
//			if (res != global_header_size) {
//				throw std::runtime_error("npz_save: header read error while adding to existing zip");
//			}
//			fseek(fp, global_header_offset, SEEK_SET);
//		}
//		else {
//			fp = fopen(zipname.c_str(), "wb");
//		}
//
//		std::vector<char> npy_header = create_npy_header(data, shape, ndims);
//
//		unsigned long nels = 1;
//		for (int m = 0; m<ndims; m++) nels *= shape[m];
//		int nbytes = nels * sizeof(T) + npy_header.size();
//
//		//get the CRC of the data to be added
//		unsigned int crc = crc32(0L, (unsigned char*)&npy_header[0], npy_header.size());
//		crc = crc32(crc, (unsigned char*)data, nels * sizeof(T));
//
//		//build the local header
//		std::vector<char> local_header;
//		local_header += "PK"; //first part of sig
//		local_header += (unsigned short)0x0403; //second part of sig
//		local_header += (unsigned short)20; //min version to extract
//		local_header += (unsigned short)0; //general purpose bit flag
//		local_header += (unsigned short)0; //compression method
//		local_header += (unsigned short)0; //file last mod time
//		local_header += (unsigned short)0;     //file last mod date
//		local_header += (unsigned int)crc; //crc
//		local_header += (unsigned int)nbytes; //compressed size
//		local_header += (unsigned int)nbytes; //uncompressed size
//		local_header += (unsigned short)fname.size(); //fname length
//		local_header += (unsigned short)0; //extra field length
//		local_header += fname;
//
//		//build global header
//		global_header += "PK"; //first part of sig
//		global_header += (unsigned short)0x0201; //second part of sig
//		global_header += (unsigned short)20; //version made by
//		global_header.insert(global_header.end(), local_header.begin() + 4, local_header.begin() + 30);
//		global_header += (unsigned short)0; //file comment length
//		global_header += (unsigned short)0; //disk number where file starts
//		global_header += (unsigned short)0; //internal file attributes
//		global_header += (unsigned int)0; //external file attributes
//		global_header += (unsigned int)global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
//		global_header += fname;
//
//		//build footer
//		std::vector<char> footer;
//		footer += "PK"; //first part of sig
//		footer += (unsigned short)0x0605; //second part of sig
//		footer += (unsigned short)0; //number of this disk
//		footer += (unsigned short)0; //disk where footer starts
//		footer += (unsigned short)(nrecs + 1); //number of records on this disk
//		footer += (unsigned short)(nrecs + 1); //total number of records
//		footer += (unsigned int)global_header.size(); //nbytes of global headers
//		footer += (unsigned int)(global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
//		footer += (unsigned short)0; //zip file comment length
//
//										//write everything      
//		fwrite(&local_header[0], sizeof(char), local_header.size(), fp);
//		fwrite(&npy_header[0], sizeof(char), npy_header.size(), fp);
//		fwrite(data, sizeof(T), nels, fp);
//		fwrite(&global_header[0], sizeof(char), global_header.size(), fp);
//		fwrite(&footer[0], sizeof(char), footer.size(), fp);
//		fclose(fp);
//	}
//
//	template<typename T> std::vector<char> create_npy_header(const T* data, const unsigned int* shape, const unsigned int ndims) {
//
//		std::vector<char> dict;
//		dict += "{'descr': '";
//		dict += BigEndianTest();
//		dict += map_type(typeid(T));
//		dict += tostring(sizeof(T));
//		dict += "', 'fortran_order': False, 'shape': (";
//		dict += tostring(shape[0]);
//		for (int i = 1;i < ndims;i++) {
//			dict += ", ";
//			dict += tostring(shape[i]);
//		}
//		if (ndims == 1) dict += ",";
//		dict += "), }";
//		//pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
//		int remainder = 16 - (10 + dict.size()) % 16;
//		dict.insert(dict.end(), remainder, ' ');
//		dict.back() = '\n';
//
//		std::vector<char> header;
//		header += (char)0x93;
//		header += "NUMPY";
//		header += (char)0x01; //major version of numpy format
//		header += (char)0x00; //minor version of numpy format
//		header += (unsigned short)dict.size();
//		header.insert(header.end(), dict.begin(), dict.end());
//
//		return header;
//	}
//
//	char BigEndianTest() {
//		unsigned char x[] = { 1,0 };
//		short y = *(short*)x;
//		return y == 1 ? '<' : '>';
//	}
//
//	char map_type(const std::type_info& t)
//	{
//		if (t == typeid(float)) return 'f';
//		if (t == typeid(double)) return 'f';
//		if (t == typeid(long double)) return 'f';
//
//		if (t == typeid(int)) return 'i';
//		if (t == typeid(char)) return 'i';
//		if (t == typeid(short)) return 'i';
//		if (t == typeid(long)) return 'i';
//		if (t == typeid(long long)) return 'i';
//
//		if (t == typeid(unsigned char)) return 'u';
//		if (t == typeid(unsigned short)) return 'u';
//		if (t == typeid(unsigned long)) return 'u';
//		if (t == typeid(unsigned long long)) return 'u';
//		if (t == typeid(unsigned int)) return 'u';
//
//		if (t == typeid(bool)) return 'b';
//
//		if (t == typeid(std::complex<float>)) return 'c';
//		if (t == typeid(std::complex<double>)) return 'c';
//		if (t == typeid(std::complex<long double>)) return 'c';
//
//		else return '?';
//	}
//
//	template<> std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs) {
//		lhs.insert(lhs.end(), rhs.begin(), rhs.end());
//		return lhs;
//	}
//
//	template<> std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs) {
//		//write in little endian
//		size_t len = strlen(rhs);
//		lhs.reserve(len);
//		for (size_t byte = 0; byte < len; byte++) {
//			lhs.push_back(rhs[byte]);
//		}
//		return lhs;
//	}
//
//	void parse_npy_header(FILE* fp, unsigned int& word_size, unsigned int*& shape, unsigned int& ndims, bool& fortran_order) {
//		char buffer[256];
//		size_t res = fread(buffer, sizeof(char), 11, fp);
//		if (res != 11)
//			throw std::runtime_error("parse_npy_header: failed fread");
//		std::string header = fgets(buffer, 256, fp);
//		assert(header[header.size() - 1] == '\n');
//
//		int loc1, loc2;
//
//		//fortran order
//		loc1 = header.find("fortran_order") + 16;
//		fortran_order = (header.substr(loc1, 5) == "True" ? true : false);
//
//		//shape
//		loc1 = header.find("(");
//		loc2 = header.find(")");
//		std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
//		if (str_shape[str_shape.size() - 1] == ',') ndims = 1;
//		else ndims = std::count(str_shape.begin(), str_shape.end(), ',') + 1;
//		shape = new unsigned int[ndims];
//		for (unsigned int i = 0;i < ndims;i++) {
//			loc1 = str_shape.find(",");
//			shape[i] = atoi(str_shape.substr(0, loc1).c_str());
//			str_shape = str_shape.substr(loc1 + 1);
//		}
//
//		//endian, word size, data type
//		//byte order code | stands for not applicable. 
//		//not sure when this applies except for byte array
//		loc1 = header.find("descr") + 9;
//		bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
//		assert(littleEndian);
//
//		//char type = header[loc1+1];
//		//assert(type == map_type(T));
//
//		std::string str_ws = header.substr(loc1 + 2);
//		loc2 = str_ws.find("'");
//		word_size = atoi(str_ws.substr(0, loc2).c_str());
//	}
//
//	void parse_zip_footer(FILE* fp, unsigned short& nrecs, unsigned int& global_header_size, unsigned int& global_header_offset)
//	{
//		std::vector<char> footer(22);
//		fseek(fp, -22, SEEK_END);
//		size_t res = fread(&footer[0], sizeof(char), 22, fp);
//		if (res != 22)
//			throw std::runtime_error("parse_zip_footer: failed fread");
//
//		unsigned short disk_no, disk_start, nrecs_on_disk, comment_len;
//		disk_no = *(unsigned short*)&footer[4];
//		disk_start = *(unsigned short*)&footer[6];
//		nrecs_on_disk = *(unsigned short*)&footer[8];
//		nrecs = *(unsigned short*)&footer[10];
//		global_header_size = *(unsigned int*)&footer[12];
//		global_header_offset = *(unsigned int*)&footer[16];
//		comment_len = *(unsigned short*)&footer[20];
//
//		assert(disk_no == 0);
//		assert(disk_start == 0);
//		assert(nrecs_on_disk == nrecs);
//		assert(comment_len == 0);
//	}
//
//	NpyArray load_the_npy_file(FILE* fp) {
//		unsigned int* shape;
//		unsigned int ndims, word_size;
//		bool fortran_order;
//		cnpy::parse_npy_header(fp, word_size, shape, ndims, fortran_order);
//		unsigned long long size = 1; //long long so no overflow when multiplying by word_size
//		for (unsigned int i = 0;i < ndims;i++) size *= shape[i];
//
//		cnpy::NpyArray arr;
//		arr.word_size = word_size;
//		arr.shape = std::vector<unsigned int>(shape, shape + ndims);
//		delete[] shape;
//		arr.data = new char[size*word_size];
//		arr.fortran_order = fortran_order;
//		size_t nread = fread(arr.data, word_size, size, fp);
//		if (nread != size)
//			throw std::runtime_error("load_the_npy_file: failed fread");
//		return arr;
//	}
//
//	npz_t npz_load(std::string fname) {
//		FILE* fp = fopen(fname.c_str(), "rb");
//
//		if (!fp) printf("npz_load: Error! Unable to open file %s!\n", fname.c_str());
//		assert(fp);
//
//		npz_t arrays;
//
//		while (1) {
//			std::vector<char> local_header(30);
//			size_t headerres = fread(&local_header[0], sizeof(char), 30, fp);
//			if (headerres != 30)
//				throw std::runtime_error("npz_load: failed fread");
//
//			//if we've reached the global header, stop reading
//			if (local_header[2] != 0x03 || local_header[3] != 0x04) break;
//
//			//read in the variable name
//			unsigned short name_len = *(unsigned short*)&local_header[26];
//			std::string varname(name_len, ' ');
//			size_t vname_res = fread(&varname[0], sizeof(char), name_len, fp);
//			if (vname_res != name_len)
//				throw std::runtime_error("npz_load: failed fread");
//
//			//erase the lagging .npy
//			varname.erase(varname.end() - 4, varname.end());
//
//			//read in the extra field
//			unsigned short extra_field_len = *(unsigned short*)&local_header[28];
//			if (extra_field_len > 0) {
//				std::vector<char> buff(extra_field_len);
//				size_t efield_res = fread(&buff[0], sizeof(char), extra_field_len, fp);
//				if (efield_res != extra_field_len)
//					throw std::runtime_error("npz_load: failed fread");
//			}
//
//			arrays[varname] = load_the_npy_file(fp);
//		}
//
//		fclose(fp);
//		return arrays;
//	}
//
//	NpyArray npz_load(std::string fname, std::string varname) {
//		FILE* fp = fopen(fname.c_str(), "rb");
//
//		if (!fp) {
//			printf("npz_load: Error! Unable to open file %s!\n", fname.c_str());
//			abort();
//		}
//
//		while (1) {
//			std::vector<char> local_header(30);
//			size_t header_res = fread(&local_header[0], sizeof(char), 30, fp);
//			if (header_res != 30)
//				throw std::runtime_error("npz_load: failed fread");
//
//			//if we've reached the global header, stop reading
//			if (local_header[2] != 0x03 || local_header[3] != 0x04) break;
//
//			//read in the variable name
//			unsigned short name_len = *(unsigned short*)&local_header[26];
//			std::string vname(name_len, ' ');
//			size_t vname_res = fread(&vname[0], sizeof(char), name_len, fp);
//			if (vname_res != name_len)
//				throw std::runtime_error("npz_load: failed fread");
//			vname.erase(vname.end() - 4, vname.end()); //erase the lagging .npy
//
//													   //read in the extra field
//			unsigned short extra_field_len = *(unsigned short*)&local_header[28];
//			fseek(fp, extra_field_len, SEEK_CUR); //skip past the extra field
//
//			if (vname == varname) {
//				NpyArray array = load_the_npy_file(fp);
//				fclose(fp);
//				return array;
//			}
//			else {
//				//skip past the data
//				unsigned int size = *(unsigned int*)&local_header[22];
//				fseek(fp, size, SEEK_CUR);
//			}
//		}
//
//		fclose(fp);
//		printf("npz_load: Error! Variable name %s not found in %s!\n", varname.c_str(), fname.c_str());
//		abort();
//	}
//
//	NpyArray npy_load(std::string fname) {
//
//		FILE* fp = fopen(fname.c_str(), "rb");
//
//		if (!fp) {
//			printf("npy_load: Error! Unable to open file %s!\n", fname.c_str());
//			abort();
//		}
//
//		NpyArray arr = load_the_npy_file(fp);
//
//		fclose(fp);
//		return arr;
//	}
//}
//
//#include "Helpers.ipp"
//
//namespace feynman {
//	
//	typedef union {
//		uint8_t b[4];
//		float f;
//	} bfloat;
//
//	Array2D<float> loadPython(const std::string &filename) 
//	{
//		cnpy::NpyArray data;
//		if (false) {
//			const cnpy::npz_t dataArray = cnpy::npz_load(filename);
//			const std::string arrayName = dataArray.begin()->first; // get the key
//			std::cout << "INFO: loadPython: found array with name " << arrayName << std::endl;
//			data = dataArray.begin()->second; // get the value
//		}
//		else {
//			data = cnpy::npy_load(filename);
//		}
//
//		if (data.fortran_order) {
//			std::cout << "WARNING: loadPython: not expecting Fortran order" << std::endl;
//			return Array2D<float>({ 0,0 });
//		}
//		if (data.word_size != 4) {
//			std::cout << "WARNING: loadPython: expecting word size 4, but got " << data.word_size << std::endl;
//			return Array2D<float>({ 0,0 });
//		}
//		if (data.shape.size() != 2) {
//			std::cout << "WARNING: loadPython: expecting 2D array, but got " << data.shape.size()  <<" dimensions." << std::endl;
//			return Array2D<float>({ 0,0 });
//		}
//		const unsigned int nColumns = data.shape[0];
//		const unsigned int nRows = data.shape[1];
//
//		std::cout << "INFO: loadPython: found 2D array: nRows=" << nRows << "; nColumns=" << nColumns << std::endl;
//		Array2D<float> result = Array2D<float>({ static_cast<int>(nRows), static_cast<int>(nColumns) });
//
//		int pos = 0;
//		for (unsigned int y = 0; y < nColumns; ++y) {
//			for (unsigned int x = 0; x < nRows; ++x) {
//								
//				bfloat tmp;
//				tmp.b[0] = data.data[pos + 0];
//				tmp.b[1] = data.data[pos + 1];
//				tmp.b[2] = data.data[pos + 2];
//				tmp.b[3] = data.data[pos + 3];
//
//				result.set(x, y, tmp.f);
//				pos += 4;
//			}
//		}
//		return result;
//	}
//}
