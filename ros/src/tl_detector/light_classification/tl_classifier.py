from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy

""" Load frozen inference graph; get scores and classes -> 
define threshold (e.g. 0.5); if score > threshold -> return class
from class number return TrafficLight attribute (e.g. TrafficLight.GREEN)
"""

MIN_SCORE_THRESHOLD = 0.5

class TLClassifier(object):
    def __init__(self):
        
        path_graph = "light_classification/model/tl_model.pb"
        self.graph = tf.Graph()

        # basically boilerplate code
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_graph, "rb") as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            # maybe not all are needed
            self.image_tensor = self.graph.get_tensor_by_name("image_tensor:0")
            self.boxes = self.graph.get_tensor_by_name("detection_boxes:0")  # bounding boxes: coordinates of detected b-boxes
            self.scores = self.graph.get_tensor_by_name("detection_scores:0")  # class scores of detections (list of list with one element)
            self.classes = self.graph.get_tensor_by_name("detection_classes:0")  # classes of detections (list of list with one element)
            self.num_detections = self.graph.get_tensor_by_name("num_detections:0")  # number of detected b-boxes in image

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            image_exp = np.expand_dims(image, axis=0)
            (_, scores, classes, _) = \
                    self.sess.run([self.boxes, self.scores, self.classes, self.num_detections], feed_dict={self.image_tensor: image_exp})

        # get rid of the unnecessary dimensions
        # boxes = boxes[0]  # not needed
        scores = scores[0]
        classes = classes[0]
        
        # The list is sorted, therefore the first score is the highest
        if scores[0] > MIN_SCORE_THRESHOLD:
            if classes[0] == 1:
                #rospy.logwarn("SCORE = {0} || DETECTED TRAFFIC LIGHT = {1}".format(scores[0], "GREEN"))
                return TrafficLight.GREEN
            elif classes[0] == 2:
                #rospy.logwarn("SCORE = {0} || DETECTED TRAFFIC LIGHT = {1}".format(scores[0], "RED"))
                return TrafficLight.RED
            elif classes[0] == 3:
                #rospy.logwarn("SCORE = {0} || DETECTED TRAFFIC LIGHT = {1}".format(scores[0], "YELLOW"))
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
