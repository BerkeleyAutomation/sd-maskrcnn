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
#include "threading.h"
#include "util.h"

Timer & ThreadedTimer::get() {
	return timer_[ std::this_thread::get_id() ];
}
ThreadedTimer::~ThreadedTimer() {
	print();
}
void ThreadedTimer::tic() {
	std::lock_guard<std::mutex> lock( m_ );
	get().tic();
}
void ThreadedTimer::toc(std::string name ) {
	std::lock_guard<std::mutex> lock( m_ );
	get().toc( name );
}
void ThreadedTimer::print() {
	std::unordered_map<std::string,double> tot;
	for( auto & t: timer_ )
		for (const auto & e: t.second.time_ )
			tot[e.first] += e.second;
	for( auto i: tot )
		printf("%20s \t %f\n", i.first.c_str(), i.second );
}
ThreadedTimer::SingleTimer::SingleTimer() : Timer() {
	print_on_exit_ = false;
}
ThreadedMemoryPool::ThreadedMemoryPool(int size) : size_(size) {
}
char *ThreadedMemoryPool::data() {
	std::lock_guard<std::mutex> lock( m_ );
	if( !pool_[ std::this_thread::get_id() ] )
		pool_[std::this_thread::get_id()] = std::make_shared<MemoryPool>( size_ );
	return pool_[ std::this_thread::get_id() ]->data();
}
