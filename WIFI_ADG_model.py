import tensorflow as tf
import first_basic_cnn as bcnn_1
import second_basic_cnn as bcnn_2
import third_basic_cnn as bcnn_3
import WIFI_ADG_ae as bae
from decorator2 import lazy_property


class ATN:
    """
    The ATN framework.
    """

    def __init__(self, data, label_gt_1, label_gt_2,label_gt_3, p_keep, rerank):
        with tf.variable_scope('autoencoder'):
            self._autoencoder = bae.BasicAE(data)
        with tf.variable_scope('target1') as scope1:
            self._target_adv1 = bcnn_1.FirstBasicCnn(
                self._autoencoder.prediction, label_gt_1, p_keep
            )
            scope1.reuse_variables()
            self._target1 = bcnn_1.FirstBasicCnn(data, label_gt_1, p_keep)    
        with tf.variable_scope('target2') as scope2:
            self._target_adv2 = bcnn_2.SecondBasicCnn(
                self._autoencoder.prediction, label_gt_2, p_keep
            )
            scope2.reuse_variables()
            self._target2 = bcnn_2.SecondBasicCnn(data, label_gt_2, p_keep)   
        with tf.variable_scope('target3') as scope3:
            self._target_adv3 = bcnn_3.ThirdBasicCnn(
                self._autoencoder.prediction, label_gt_3, p_keep
            )
            scope3.reuse_variables()
            self._target3 = bcnn_3.ThirdBasicCnn(data, label_gt_3, p_keep)   
            
            #self._model1 = bcnn1.BasicCnn(data, label_gt, p_keep)
        self.data = data
        self.rerank = rerank
        self.label_gt_1 = label_gt_1
        self.label_gt_2 = label_gt_2
        self.label_gt_3 = label_gt_3
        self.prediction
        self.optimization

    @lazy_property
    def optimization(self):
        loss_beta = 1 #beta,代码中给0.1,但是文章中远没有这么高，0.001-0.01
        learning_rate = 0.0001

        x_pred = self._autoencoder.prediction
        x_true = self.data

        L0 = tf.reduce_sum( # beta*L2(x,x')
            tf.sqrt(tf.reduce_sum((x_pred-x_true)**2))
        )/70
        
        loss_act = tf.reduce_mean(-tf.reduce_sum(self.label_gt_1 * 
        tf.log(tf.clip_by_value(self._target_adv1.prediction,1e-10,
        tf.reduce_max(self._target_adv1.prediction))), reduction_indices=[1]))
        new_act = tf.reduce_mean(-tf.reduce_sum((1-self.label_gt_1) * 
        tf.log(tf.clip_by_value(self._target_adv1.prediction,1e-10,
        tf.reduce_max(self._target_adv1.prediction))), reduction_indices=[1]))
        
        loss_per = tf.reduce_mean(-tf.reduce_sum(self.label_gt_2 * 
        tf.log(tf.clip_by_value(self._target_adv2.prediction,1e-10,
        tf.reduce_max(self._target_adv2.prediction))), reduction_indices=[1]))
        new_per = tf.reduce_mean(-tf.reduce_sum((1-self.label_gt_2) * 
        tf.log(tf.clip_by_value(self._target_adv2.prediction,1e-10,
        tf.reduce_max(self._target_adv2.prediction))), reduction_indices=[1]))
        
        loss_loc = tf.reduce_mean(-tf.reduce_sum(self.label_gt_3 * 
        tf.log(tf.clip_by_value(self._target_adv3.prediction,1e-10,
        tf.reduce_max(self._target_adv3.prediction))), reduction_indices=[1]))
        new_loc = tf.reduce_mean(-tf.reduce_sum((1-self.label_gt_3) * 
        tf.log(tf.clip_by_value(self._target_adv3.prediction,1e-10,
        tf.reduce_max(self._target_adv3.prediction))), reduction_indices=[1]))
        
        loss =  loss_act+loss_per-loss_loc
         
        optimizer_ae = tf.train.AdamOptimizer(learning_rate).minimize(
            loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "autoencoder"))
#        optimizer_act = tf.train.AdamOptimizer(learning_rate).minimize(
#            loss_act,
#            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
#                                       "target1"))
#        optimizer_per = tf.train.AdamOptimizer(learning_rate).minimize(
#            loss_per,
#            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
#                                       "target2"))
        return optimizer_ae, loss,L0,loss_act, loss_per,loss_loc

    @lazy_property
    def prediction(self):
        return self._autoencoder.prediction

    def load_ae(self, sess, path, prefix="ATN_"):
        self._autoencoder.load(sess, path+'/AE_CSI', name=prefix+'basic_ae.ckpt')
        
    def load_model(self, sess, path, prefix="ATN_"):
        self._target1.load(sess, path+'/C_act5')
        self._target2.load(sess, path+'/C_id5')
        self._target3.load(sess, path+'/C_room5')

    def save_ae(self, sess, path, prefix="ATN_"):
        self._autoencoder.save(sess, path+'/AE_CSI', name=prefix+'basic_ae.ckpt')
        
    def save_model(self, sess, path, prefix="ATN_"):
#        self._autoencoder.save(sess, path+'/AE_CSI', name=prefix+'basic_ae.ckpt')
        self._target1.save(sess, path+'/C_act5')
        self._target2.save(sess, path+'/C_id5')
        self._target3.save(sess, path+'/C_room5')
