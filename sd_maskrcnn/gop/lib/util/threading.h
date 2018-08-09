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
#pragma once
#include "util.h"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <atomic>
#include <deque>
#include <vector>
#include <memory>

template<typename T> class LockFreeQueue {
public:
	enum Status {
		SUCCESS=0,
		EMPTY=-1,
		FULL=-2,
	};
protected:
	const size_t size_;
	std::atomic<int> s_, ws_, e_, we_;
	std::vector<T> v_;
public:
	LockFreeQueue( size_t size=0 ): size_(size), s_(0), ws_(0), e_(0), we_(0), v_( 2*size+5 ) {
	}
	Status try_push( const T & v ) {
		if( we_-ws_ >= size_ )
			return FULL;
		int e = we_++;
		v_[e % v_.size()] = v;
		// Update e_
		while( 1 ) {
			int ee = e;
			if( e_.compare_exchange_weak( ee, e+1 ) )
				break;
		}
		return SUCCESS;
	}
	Status try_pop( T & v ) {
		// Get an element id
		int s = s_;
		while(1) {
			if( e_<=s )
				return EMPTY;
			if( s_.compare_exchange_weak( s, s+1 ) )
				break;
		}
		// Read
		v = v_[s % v_.size()];
		// and mark it as read
		while( 1 ) {
			int ss = s;
			if( ws_.compare_exchange_weak( ss, s+1 ) )
				break;
		}
		return SUCCESS;
	}
};

template<typename T>
class ThreadedQueue {
public:
	class Queue {
	protected:
		friend class ThreadedQueue;
		typedef std::function<void ( const T & )> Enqueue;
		Enqueue f_;
		Queue( const Enqueue & f ):f_(f) {}
	public:
		void push( const T & t ) {
			f_( t );
		}
	};
	typedef std::function<void (Queue*, T)> F;
protected:
	typedef LockFreeQueue<T> LFQ;
	LFQ q_;
	std::mutex m_;
	std::condition_variable cv_;
	std::atomic<int> running_;
	
	bool wait_pop( T & t ) {
		// Wait for an element on the global queue
		const int WAIT_TIME = 500;//microseconds
		typename LFQ::Status s = q_.try_pop( t );
		while( s != LFQ::SUCCESS && running_ ) {
			if( !running_ )
				return false;
			// Wait for work
			{
				std::unique_lock<std::mutex> lock(m_);
				cv_.wait_for(lock,std::chrono::milliseconds(WAIT_TIME));
			}
			s = q_.try_pop( t );
		}
		if( !running_ )
			return false;
		return true;
	}
	bool push( const T& v ) {
		// Add an element to the global queue
		typename LFQ::Status s = q_.try_push( v );
		if( s == LFQ::SUCCESS ) {
			cv_.notify_one();
			return true;
		}
		return false;
	}
	void stop() {
		// Stop all threads
		running_ = 0;
		cv_.notify_all();
	}
	void work( F f ) {
		std::deque<T> q;
		Queue f_q( [&](const T&t){ running_++; if( !push(t) ) q.push_back(t); } );
		while(1)
		{
			// Get a task
			T t;
			if( !q.empty() ) {
				t = q.front();
				q.pop_front();
				// Give some tasks away ..
				while( !q.empty() && push(t) ) {
					// .. and fetch a new one
					t = q.front();
					q.pop_front();
				}
			}
			// Get the task from the global queue
			else{
				if( !wait_pop( t ) ) {
					stop();
					return;
				}
			}
			// Call the function
			f( &f_q, t );
			// Mark it as done
			running_--;
		}
	}
public:
	ThreadedQueue():q_(std::thread::hardware_concurrency()),running_(0){
	}
	void process( F f, const std::vector<T> & data ) {
		// Add as many elements as we can
		running_ = data.size();
		size_t k = 0;
		while( k < data.size() && q_.try_push( data[k] ) == LFQ::SUCCESS )
			k++;
		
		// Spawn the workers
		const int NT = std::thread::hardware_concurrency();
		std::vector< std::thread > workers;
		for(size_t i = 0;i<NT;i++)
			workers.push_back( std::thread( [&](){work(f);} ) );
		
		// Add the rest of the work
		for( ; k<data.size(); k++ ) {
			while( !push( data[k] ) )
				std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
		}
		
		// Merge the workers
		for(size_t i = 0;i<workers.size();++i)
			workers[i].join();
	}
	void process( F f, const T & data ) {
		process( f, std::vector<T>(1,data) );
	}
};

template<typename T>
void update_maximum(std::atomic<T>& maximum_value, const T & value)
{
    T prev_value = maximum_value;
    while(prev_value < value && !maximum_value.compare_exchange_weak(prev_value, value));
}

struct ThreadedTimer {
	struct SingleTimer: public Timer {
		SingleTimer();
	};
	std::mutex m_;
	std::unordered_map<std::thread::id,SingleTimer> timer_;
	Timer & get();
	
	~ThreadedTimer();
	void tic();
	void toc(std::string name="other");
	void print();
};
class ThreadedMemoryPool {
	std::mutex m_;
	int size_;
	std::unordered_map< std::thread::id, std::shared_ptr<MemoryPool> > pool_;
	MemoryPool & get();
public:
	ThreadedMemoryPool( int size );
	char * data();
};
