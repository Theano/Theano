import numpy as np
from googlenet_theano import googlenet, compile_train_model, compile_val_model, set_learning_rate
import time
from datetime import datetime
import traceback


try:
    import theano.sandbox.mlsl.multinode as distributed
    print('mlsl is imported')
except ImportError as e:
    print ('Failed to import distributed module, please double check')
    print(traceback.format_exc())

def time_theano_run(func, info_string):
    num_batches = 50
    num_steps_burn_in = 10
    durations = []
    for i in xrange(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = func()
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: Iteration %d, %s, time: %.2f ms' %
                      (datetime.now(), i - num_steps_burn_in, info_string, duration * 1000))
            durations.append(duration)
    durations = np.array(durations)
    print('%s: Average %s pass: %.2f ms ' %
          (datetime.now(), info_string, durations.mean() * 1000))


def googlenet_train(batch_size=256, image_size=(3, 224, 224)):
    input_shape = (batch_size,) + image_size
    model = googlenet(input_shape, drop_flag=True)

    dist = distributed.Distribution()
    print ('dist.rank: ', dist.rank, 'dist.size: ', dist.size)

    distributed.set_global_batch_size(batch_size * dist.size)
    distributed.set_param_count(len(model.params))

    (train_model, shared_x, shared_y, shared_lr) = compile_train_model(model, batch_size = batch_size)

    #if dist.rank == 0:
    #    print("Print Model")
    #    theano.printing.debugprint(train_model)    

    (validate_model, shared_x, shared_y) = compile_val_model(model, batch_size = batch_size)

    images = np.random.random_integers(0, 255, input_shape).astype('float32')
    labels = np.random.random_integers(0, 999, batch_size).astype('int32')
    shared_x.set_value(images)
    shared_y.set_value(labels)

    print("Run model validation")
    model.set_dropout_off()
    time_theano_run(validate_model, 'Forward')
    print("Run model training")
    model.set_dropout_on()
    time_theano_run(train_model, 'Forward-Backward')

    dist.destroy()

if __name__  == '__main__':
    googlenet_train(batch_size=32)
