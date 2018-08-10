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
#include "util.h"
#include <chrono>

double getTime(){
	static auto t0 = std::chrono::system_clock::now();
	return std::chrono::duration_cast< std::chrono::duration<double> >(std::chrono::system_clock::now()-t0).count() * 1000.;
}
static double tic_time = 0;
void tic(){
	tic_time = getTime();
}
double toc(){
	return getTime() - tic_time;
}
double toc( const char * str ){
	double t = toc();
	printf( str, t );
	return t;
}
Clock::Clock():tic_time( getTime() ) {
	
}
void Clock::tic() {
	tic_time = getTime();
}
double Clock::toc() {
	double r = getTime() - tic_time;
	tic_time = getTime();
	return r;
}
Timer::Timer():Clock(),print_on_exit_(true) {
}
Timer::~Timer() {
	if( print_on_exit_ ) {
		toc();
		print();
	}
}
void Timer::toc(std::string name) {
	time_[name] += Clock::toc();
}
void Timer::print() const {
	for( auto i: time_ )
		printf("%20s \t %f\n", i.first.c_str(), i.second );
}
ProgressPrint::ProgressPrint( const std::string & msg, float start, float end ) : msg_( msg ), start_(start), range_(end-start) {
	update(start_);
}
ProgressPrint::ProgressPrint( const std::string &msg, float end ) : msg_( msg ), start_(0), range_(end) {
	update(start_);
}
ProgressPrint::~ProgressPrint() {
	update( start_ + range_ );
	printf( "\n" );
}
void ProgressPrint::update( float n ) const {
	printf("%s[%0.1f%%]\r", msg_.c_str(), 100*(n-start_)/range_ );
	fflush( stdout );
}
void UpdateProgressPrint::updateDelta( float delta_n ) {
	const float min_delta = 0.1;
	float p = p_;
	while( !p_.compare_exchange_weak(p, p+delta_n) );
	float dp = ((int)(1000*p+delta_n) - (int)(1000*p))/1000.f;
	if( dp >= min_delta )
		ProgressPrint::update( p_+delta_n );
}
MemoryPool::MemoryPool(int size) : data_(new char[size]) {
}
MemoryPool::~MemoryPool() {
	delete [] data_;
}
char *MemoryPool::data() const {
	return data_;
}
