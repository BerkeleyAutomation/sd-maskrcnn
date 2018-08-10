/*
    Copyright (c) 2014, Philipp Krähenbühl
    All rights reserved.
	
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.
	
    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "voc.h"
#include "imgproc/image.h"
#include "apng.h"
#include <string>
#include <fstream>
#include <unordered_map>
#include <rapidxml.hpp>

#define XSTR( x ) STR( x )
#define STR( x ) std::string( #x )
#ifdef VOC_DIR
std::string voc_dir = XSTR(VOC_DIR);
#else
std::string voc_dir = ".";
#endif
const std::string VOC_IMAGES = "/JPEGImages/%s.jpg";
const std::string VOC_CLASS = "/SegmentationClass/%s.png";
const std::string VOC_OBJECT = "/SegmentationObject/%s.png";
const std::string VOC_ANNOT = "/Annotations/%s.xml";

np::ndarray cleanVOC( const np::ndarray& lbl ) {
	unsigned char * plbl = (unsigned char *)lbl.get_data();
	np::ndarray r = np::zeros( lbl.get_nd(), lbl.get_shape(), np::dtype::get_builtin<short>() );
	short * pr = (short *)r.get_data();
	for( int i=0; i<lbl.shape(0)*lbl.shape(1); i++ )
		pr[i] = (plbl[i]>250)?-2:(short)plbl[i]-1;
	return r;
}
static list readAnnotation( const std::string & annot ) {
	list r;
	using namespace rapidxml;
	xml_document<> doc;    // character type defaults to char
	std::ifstream t(annot);
	std::string xml_str = std::string(std::istreambuf_iterator<char>(t),std::istreambuf_iterator<char>());
	doc.parse<0>( (char*)xml_str.c_str() );
	
	xml_node<> *annotation = doc.first_node("annotation");
	for( xml_node<> *objects = annotation->first_node(); objects; objects = objects->next_sibling() ) 
		if( objects->name() == std::string("object") ){
			dict o;
			o["name"] = std::string( objects->first_node("name")->value() );
			o["difficult"] = (bool)std::stoi( std::string( objects->first_node("difficult")->value() ) );
			xml_node<> * bbox = objects->first_node("bndbox");
			list bblist;
			bblist.append(std::stoi( std::string( bbox->first_node("xmin")->value() ) ));
			bblist.append(std::stoi( std::string( bbox->first_node("ymin")->value() ) ));
			bblist.append(std::stoi( std::string( bbox->first_node("xmax")->value() ) ));
			bblist.append(std::stoi( std::string( bbox->first_node("ymax")->value() ) ));
			o["bbox"] = bblist;
			
			r.append( o );
		}
	return r;
}
template<int YEAR>
static dict loadEntry( const std::string & name, bool load_seg = true, bool load_im = true ) {
	const std::string base_dir = voc_dir + "/VOC" + std::to_string(YEAR) + "/";
	dict r;
	char buf[1024];
	if( load_seg ) {
		sprintf( buf, (base_dir+VOC_OBJECT).c_str(), name.c_str() );
		np::ndarray olbl = readIPNG( buf );
		if( !olbl.get_nd() )
			return dict();
		r["segmentation"] = cleanVOC(olbl);
		
		sprintf( buf, (base_dir+VOC_CLASS).c_str(), name.c_str() );
		np::ndarray clbl = readIPNG( buf );
		if( !clbl.get_nd() )
			return dict();
		r["class"] = cleanVOC(clbl);
	}
	if (load_im) {
		sprintf( buf, (base_dir+VOC_IMAGES).c_str(), name.c_str() );
		std::shared_ptr<Image8u> im = imreadShared( buf );
		if( !im || im->empty() )
			return dict();
		r["image"] = im;
	}
	sprintf( buf, (base_dir+VOC_ANNOT).c_str(), name.c_str() );
	r["annotation"] = readAnnotation( buf );
	
	r["name"] = name;
	return r;
}
list loadVOC2012_detect( bool train, bool valid, bool test ) {
	const std::string VOC_2012_DIR = voc_dir + "/VOC2012/";
	bool read[3]={train,valid,test};
	std::string fn[3]={"ImageSets/Main/train.txt","","ImageSets/Main/val.txt"};
	list r;
	for( int i=0; i<3; i++ ) 
		if( read[i] ){
			std::ifstream is(VOC_2012_DIR+"/"+fn[i]);
			while(is.is_open() && !is.eof()) {
				std::string l;
				std::getline(is,l);
				if( !l.empty() ) {
					dict d = loadEntry<2012>( l, false );
					if( len( d ) )
						r.append( d );
				}
			}
		}
	return r;
}
list loadVOC2012( bool train, bool valid, bool test ) {
	const std::string VOC_2012_DIR = voc_dir + "/VOC2012/";
	bool read[3]={train,valid,test};
	std::string fn[3]={"ImageSets/Segmentation/train.txt","","ImageSets/Segmentation/val.txt"};
	list r;
	for( int i=0; i<3; i++ ) 
		if( read[i] ){
			std::ifstream is(VOC_2012_DIR+"/"+fn[i]);
			while(is.is_open() && !is.eof()) {
				std::string l;
				std::getline(is,l);
				if( !l.empty() ) {
					dict d = loadEntry<2012>( l );
					if( len( d ) )
						r.append( d );
				}
			}
		}
	return r;
}
list loadVOC2012_small( bool train, bool valid, bool test ) {
	const std::string VOC_2012_DIR = voc_dir + "/VOC2012/";
	bool read[3]={train,valid,test};
	std::string fn[3]={"ImageSets/Segmentation/train.txt","","ImageSets/Segmentation/val.txt"};
	list r;
	for( int i=0; i<3; i++ )
		if( read[i] ){
			int n=0;
			std::ifstream is(VOC_2012_DIR+"/"+fn[i]);
			while(is.is_open() && !is.eof()) {
				std::string l;
				std::getline(is,l);
				if( !l.empty() ) {
					dict d = loadEntry<2012>( l );
					if( len( d ) ) {
						n++;
						r.append( d );
					}
				}
				if( n >= 10 )
					break;
			}
		}
	return r;
}
list loadVOC2007( bool train, bool valid, bool test ) {
	const std::string VOC_2007_DIR = voc_dir + "/VOC2007/";
	bool read[3]={train,valid,test};
	std::string fn[3]={"ImageSets/Segmentation/train.txt","ImageSets/Segmentation/val.txt","ImageSets/Segmentation/test.txt"};
	list r;
	for( int i=0; i<3; i++ ) 
		if( read[i] ){
			std::ifstream is(VOC_2007_DIR+"/"+fn[i]);
			while(is.is_open() && !is.eof()) {
				std::string l;
				std::getline(is,l);
				if( !l.empty() ) {
					dict d = loadEntry<2007>( l );
					if( len( d ) )
						r.append( d );
				}
			}
		}
	return r;
}
list loadVOC2007_detect_noim( bool train, bool valid, bool test ) {
	const std::string VOC_2007_DIR = voc_dir + "/VOC2007/";
	bool read[3]={train,valid,test};
	std::string fn[3]={"ImageSets/Main/train.txt","ImageSets/Main/val.txt","ImageSets/Main/test.txt"};
	list r;
	int cnt = 0;
	for( int i=0; i<3; i++ ) 
		if( read[i] ){
			std::ifstream is(VOC_2007_DIR+"/"+fn[i]);
			while(is.is_open() && !is.eof()) {
				std::string l;
				std::getline(is,l);
				if( !l.empty() ) {
					cnt++;
					dict d = loadEntry<2007>( l, false, false );
					if( len( d ) )
						r.append( d );
				}
			}
		}
	return r;
}
list loadVOC2007_detect( bool train, bool valid, bool test ) {
	const std::string VOC_2007_DIR = voc_dir + "/VOC2007/";
	bool read[3]={train,valid,test};
	std::string fn[3]={"ImageSets/Main/train.txt","ImageSets/Main/val.txt","ImageSets/Main/test.txt"};
	list r;
	int cnt = 0;
	for( int i=0; i<3; i++ ) 
		if( read[i] ){
			std::ifstream is(VOC_2007_DIR+"/"+fn[i]);
			while(is.is_open() && !is.eof()) {
				std::string l;
				std::getline(is,l);
				if( !l.empty() ) {
					cnt++;
					dict d = loadEntry<2007>( l, false );
					if( len( d ) )
						r.append( d );
				}
			}
		}
	return r;
}
list loadVOC2010(bool train, bool valid, bool test) {
	const std::string VOC_2010_DIR = voc_dir + "/VOC2010/";
	bool read[3]={train,valid,test};
	std::string fn[3]={"ImageSets/Segmentation/train.txt","","ImageSets/Segmentation/val.txt"};
	list r;
	for( int i=0; i<3; i++ ) 
		if( read[i] ){
			std::ifstream is(VOC_2010_DIR+"/"+fn[i]);
			while(is.is_open() && !is.eof()) {
				std::string l;
				std::getline(is,l);
				if( !l.empty() ) {
					dict d = loadEntry<2010>( l );
					if( len( d ) )
						r.append( d );
				}
			}
		}
	return r;
}
