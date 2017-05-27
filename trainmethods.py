import numpy as np

from custom_keras import EarlyStoppingBound
from plottools import plot_metrics, plot_batch, plot_all


from keras.callbacks import TensorBoard, EarlyStopping

import pdb



def train_acc_bound(model_d, model_dg, model_g,
                    gen_d_train, gen_d_eval, gen_dg_train, gen_dg_eval,
                    epochs=100, bound=0.8,
                    steps_per_cycle=1, validation_batchs=50):
    # tboard_g = TensorBoard(log_dir='./logs', histogram_freq=1,
    #                        write_graph=True, write_images=True)
    # tboard_d = TensorBoard(log_dir='./logs', histogram_freq=1,
    #                        write_graph=True, write_images=True)
    bound_acc = EarlyStoppingBound(bound, 'val_binary_accuracy', 'upper')
    early_stop = EarlyStopping('val_loss', min_delta=0.001, patience=6)

    epoch_g = 0
    epoch_d = 0
    for count in range(epoch):
        print("Starting loop "+str(count))

        print("Train generative network")
        model_dg.fit_generator(gen_dg_train, steps_per_epoch=steps_per_cycle,
                               epochs = 100,
                               callbacks=[tboard_g, bound_acc, early_stop],
                               validation_data=gen_g_eval,
                               validation_steps=validation_batchs)#,


        print("Printing an image.")
        batch_seed, _ = gen_g_eval.next()
        batch_img = model_g.predict_on_batch(batch_seed)
        plot_batch(batch_img, namefile="img/gen"+str(count)+".png")



        print("Train Discriminative network")
        model_d.fit_generator(gen_d_train, steps_per_epoch=steps_per_cycle,
                              epochs = 100, workers=1,
                              callbacks=[bound_acc, early_stop],
                              validation_data=gen_d_eval,
                              validation_steps=validation_batchs)

def train_each(model_d, model_dg, model_g,
               gen_d_train, gen_d_eval, gen_dg_train, gen_dg_eval,
               train_size, batch_size, nepoch=100, 
               track_metrics=False, namefile_prefix="img/output"):
    #gamma = 1
    for epoch in range(nepoch):
        loss_dict = {"Generator":[],
                     "Discriminator":[],
                     "Discriminator (gen)":[], 
                     "Discriminator (real)":[]}
        acc_dict =  {"Generator":[],
                     "Discriminator":[],
                     "Discriminator (gen)":[], 
                     "Discriminator (real)":[]}
        print("EPOCH : "+str(epoch))
        d_scores, dg_scores = [], []
        #d_scores_eval, d_scores_train, dg_scores_eval = [], [], []
        for index in range(train_size//batch_size):
            #for index in range(60):
            if index % 20 == 1:
                print(index, train_size//batch_size)
                dloss = np.mean([x for (x,_) in d_scores[-20:]])
                dacc = np.mean([y for (_,y) in d_scores[-20:]])
                dgloss = np.mean([x for (x,_) in dg_scores[-20:]])
                dgacc = np.mean([y for (_,y) in dg_scores[-20:]])
                print("D loss : %f ; D acc : %f" % (dloss, dacc))
                print("G loss : %f ; G acc : %f" % (dgloss, dgacc))
            #     print("Discriminator loss : %f accuracy : %f" % \
            #           (d_loss / index, d_acc / index))
            #     print("Generator loss : %f accuracy : %f" % \
            #           (dg_loss / (index*gamma), dg_acc / (index*gamma)))



            X, y = gen_d_train.next()
            d_scores.append(model_d.train_on_batch(X, y))
            #if track_metrics:
            #    d_scores_eval.append(model_d.evaluate_generator(gen_d_eval,
            #                                                    steps=10))
            #    d_scores_train.append(model_d.evaluate_generator(gen_d_train,
            #                                                     steps=10))
            #    dg_scores_eval.append(model_dg.evaluate_generator(gen_dg_eval,
            #                                                      steps=10))
            #for _ in range(gamma):
            X, y = gen_dg_train.next()
            dg_scores.append(model_dg.train_on_batch(X,y))
            if track_metrics:
                loss, acc = {},{}
                gen_d_eval.set_mode('gen')
                loss["Discriminator (gen)"], acc["Discriminator (gen)"] = \
                    model_d.evaluate_generator(gen_d_eval, steps=5)
                gen_d_eval.set_mode('real')
                loss["Discriminator (real)"], acc["Discriminator (real)"] = \
                    model_d.evaluate_generator(gen_d_eval, steps=5)
                gen_d_eval.set_mode(None)
                
                loss["Generator"], acc["Generator"] = \
                    model_dg.evaluate_generator(gen_dg_eval, steps=10)
                for k in loss:
                    loss_dict[k].append(loss[k])
                    acc_dict[k].append(acc[k])

        # print("Printing an image.")
        # batch_dg, _ = gen_dg_eval.next()
        # batch_img = model_g.predict_on_batch(batch_dg)
        # plot_batch(batch_img, namefile="img/gen"+str(epoch)+".png")

        if track_metrics:
            batch_gen, _ = gen_d_eval.get_batch_gen(batch_size=144)
            batch_real, _ = gen_d_eval.get_batch_real(batch_size=144)

            score_gen = model_d.predict_on_batch(batch_gen)[:,0]
            score_real = model_d.predict_on_batch(batch_real)[:,0]
            for k in loss_dict:
                loss_dict[k] = np.array(loss_dict[k])
                acc_dict[k] = np.array(acc_dict[k])
                
            loss_dict["Discriminator"] = 1/2* \
                (loss_dict["Discriminator (gen)"] + 
                loss_dict["Discriminator (real)"])
            acc_dict["Discriminator"] = 1/2* \
                (acc_dict["Discriminator (gen)"] + 
                acc_dict["Discriminator (real)"])
            plot_all(loss_dict, acc_dict,
                     batch_real, batch_gen, 
                     score_gen=score_gen, score_true=score_real,
                     namefile= "%s.%03d.pdf" % (namefile_prefix, epoch),
                     data_per_step=batch_size//2)

            # metrics = {"Discriminator loss": [x for x,_ in d_scores_eval],
            #            "Discriminator accuracy": [y for _, y in d_scores_eval],
            #            "Generator loss":[x for x,_ in dg_scores_eval],
            #            "Generator accuracy": [y for _, y in dg_scores_eval]}
            # plot_metrics(metrics, namefile="img/metrics"+str(epoch)+".png")
        
        else:    
            print("Printing an image.")
            batch_dg, _ = gen_dg_eval.next()
            batch_img = model_g.predict_on_batch(batch_dg)
            plot_batch(batch_img, namefile="img/celeba_gen"+str(epoch)+".png")
