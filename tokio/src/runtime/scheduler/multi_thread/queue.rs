
use std::convert::TryInto;

use bwosqueue::{Owner, Stealer};
use crate::runtime::task::{self, Inject};
use crate::runtime::MetricsBatch;


// todo: Discuss using const generics or runtime values. Benchmark performance difference.
const NUM_BLOCKS: usize = 8;
const ELEMENTS_PER_BLOCK: usize = 32;

/// Producer handle. May only be used from a single thread.
pub(crate) struct Local<T: 'static> {
    inner: Owner<task::Notified<T>, NUM_BLOCKS, ELEMENTS_PER_BLOCK>,
}


/// Consumer handle. May be used from many threads.
pub(crate) struct Steal<T: 'static>(Stealer<task::Notified<T>, NUM_BLOCKS, ELEMENTS_PER_BLOCK>);

pub(crate) fn local<T: 'static>() -> (Steal<T>, Local<T>) {

    let (owner, stealer) = bwosqueue::new::<task::Notified<T>, NUM_BLOCKS, ELEMENTS_PER_BLOCK>();

    let local = Local {
        inner: owner,
    };

    let remote = Steal(stealer);

    (remote, local)
}

impl<T> Local<T> {
    /// Returns true if the queue has entries that can be stolen.
    pub(crate) fn is_stealable(&self) -> bool {
        self.inner.has_stealable_block()
    }

    /// Returns true if there are entries in the queue.
    pub(crate) fn has_tasks(&self) -> bool {
           self.inner.can_consume()
    }

    /// Pushes a task to the back of the local queue, skipping the LIFO slot.
    pub(crate) fn push_back(
        &mut self,
        task: task::Notified<T>,
        inject: &Inject<T>,
        metrics: &mut MetricsBatch,
    ) {
        if let Err(t) = self.inner.enqueue(task){
            if self.inner.has_stealers() {
                inject.push(t);
            } else {
                // push overflow of old queuue
                if let Some(block_iter) = self.inner.dequeue_block() {
                    // could use `and_then` to chain block dequeues a couple of times if
                    // successfull, if we want to steal more than one block
                    inject.push_batch(block_iter.chain(std::iter::once(t)))
                } else {
                    // Give up and use inject queue.
                    self.inner.enqueue(t).unwrap_or_else(|t| inject.push(t))
                }
            }
            // Add 1 to factor in the task currently being scheduled.
            metrics.incr_overflow_count();
        };

    }



    pub(crate) fn pop(&mut self) -> Option<task::Notified<T>> {
        self.inner.dequeue()
    }
}


impl<T> Steal<T> {
    pub(crate) fn is_empty(&self) -> bool {
        self.0.estimated_queue_entries() == 0
    }


     /// Steals one block from self and place them into `dst`.
    pub(crate) fn steal_into(
        &self,
        dst: &mut Local<T>,
        dst_metrics: &mut MetricsBatch,
        inject: &Inject<T>,
    ) -> Option<task::Notified<T>> {

        // Steal nothing strategy (testing)
        None

    }

}

impl<T> Clone for Steal<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}



cfg_metrics! {
    impl<T> Steal<T> {
        pub(crate) fn len(&self) -> usize {
            self.0.estimated_queue_entries()
        }
    }
}

impl<T> Drop for Local<T> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert!(self.pop().is_none(), "queue not empty");
        }
    }
}

