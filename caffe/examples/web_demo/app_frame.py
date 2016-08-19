import os
import math
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib2
import exifutil
from nltk.corpus import wordnet as wn
import caffe

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

#wrap the loading of the url in a timeout. No url should take longer than .5 seconds to load
import time
import signal
def timeout_catcher(signnum, _):
    raise urllib2.URLError("read timeout: max 2 sec, is the image large?")

signal.signal(signal.SIGALRM, timeout_catcher)

@app.route('/classify_url', methods=['GET'])
def classify_url():

    imageurl = flask.request.args.get('imageurl', '')
    try:
        signal.setitimer(signal.ITIMER_REAL, 2.0)
        string_buffer = StringIO.StringIO(
            urllib2.urlopen(imageurl).read())
		
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        signal.setitimer(signal.ITIMER_REAL, 0)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL ('+str(err).replace("<","").replace(">","")+'):' )
        )
    signal.setitimer(signal.ITIMER_REAL, 0)
    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/../of500_crf_1024/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/../of500_crf_1024/crf.caffemodel.h5'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'syns': (
            '{}/../of500/of500_object_id_syn.tab'.format(REPO_DIRNAME)),
        'structure': (
            '{}/../of500/of500_argvalues.tab'.format(REPO_DIRNAME)),
        'role_order': ( 
            '{}/../of500/role_order.txt'.format(REPO_DIRNAME)),
	'nn_file': (
          '{}/../of500_crf_1024/nn.npy'.format(REPO_DIRNAME)),
	'image_file': (
	  '{}/../of500_crf_1024/nn.names'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    print(default_args)
    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, syns, structure, role_order, nn_file, image_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )
        print(np.load(mean_file).mean(1).mean(1))

	f = open(syns)
	self.syn_id = {}
	for line in f.readlines():
		tabs = line.split("\t")
		self.syn_id[tabs[0].strip()] = tabs[1].strip()
	f = open(structure)
	self.id_verb = {}
        self.verb_images = {} 
	for line in f.readlines():
		tabs = line.split("\t")
		verb = tabs[0]
		if verb not in self.verb_images: self.verb_images[verb] = []
		role = tabs[1]
		verbid = int(tabs[2])
		if verbid not in self.id_verb:
			v = {}
			v["name"] = verb
			v["roles"] = {} 
			self.id_verb[verbid] = v
		v = self.id_verb[verbid]
		roleid = int(tabs[3])
                nids = int(tabs[4])
                nounids = {}
                 
		for i in range(0,nids):
 			n = tabs[5+2*i].strip()
			ix = tabs[5+2*i+1].strip()
			if n == "-1": n = "null"
			else: n = self.syn_id[n]
			nounids[int(ix)]=n
	        		
		v["roles"][roleid] = {"name":role.lower(), "values": nounids}
       
	features = np.load(nn_file)
        files = open(image_file).readlines()
        i = 0
        for f in files:
	   img = f.split("/")[-1].strip()
	   verb = img.split("_")[0]
	   self.verb_images[verb].append((img,features[i,:]))
	   i+=1
	print( "total number of vectors: " + str(i))
        f = open(role_order)
        for line in f.readlines():
          tabs = line.split("\t")
          found = ''
          for (k,v) in self.id_verb.items():
 	    if v["name"] == tabs[0]:
              for (k2, v2) in v["roles"].items():
                if v2["name"] == tabs[1]:
                  found = v2
        	  break
            if found != '' : break
          if found != '' : found["index"] = int(tabs[2])

	self.wn_map = {}
        for s in wn.all_synsets():
          if s.pos() == 'n' :
                w = [] 
                for lemma in s.lemmas():
                        w.append(lemma.name().replace("_"," "))
                self.wn_map["n{0:08d}".format(s.offset())] = w;
        #print("done with constructor");
        #with open(class_labels_file) as f:
        #    labels_df = pd.DataFrame([
        #        {
        #            'synset_id': l.strip().split(' ')[0],
        #            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        #        }
        #        for l in f.readlines()
        #    ])
        #self.labels = labels_df.sort('synset_id')['name'].values

        #self.bet = cPickle.l ad(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        #self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def classify_image(self, image):
        try:
            starttime = time.time()
            rv = self.net.predict([image], oversample=False)
            struct = rv["frame-structure"].flatten()
            score = rv["frame-score"].flatten()
            features = rv["features"].flatten()
	    print features[0:100]
            endtime = time.time()
            items = []
	    print struct 
            for i in range(0, len(score)):
               values = []
               for j in range(1,7):
	         val = struct[i*7 + j]
                 if j-1 not in self.id_verb[i]["roles"]: break 
                 #values[self.id_verb[i]["roles"][j-1]["name"]] = val
                 noun = self.id_verb[i]["roles"][j-1]["values"][int(val)]
		 if noun != "null" : noun = self.wn_map[noun][0]
                 if noun == "null" : noun = u"\u2205"
		 values.append ((self.id_verb[i]["roles"][j-1]["name"], noun, self.id_verb[i]["roles"][j-1]["index"]))
               values = sorted(values, key = lambda x: x[2]) 
               items.append((self.id_verb[i]["name"],values, math.exp(score[i]), "{0:.5f}".format(math.exp(score[i]))))
            items = sorted(items, key = lambda x: -x[2])
	    items = items[:10]
            #find the nearest nieghbors
	    row = []
            for v in items:
		images = self.verb_images[v[0]]
		output = []
		for (img,fea) in images:
			d = np.linalg.norm(features.flatten()-fea.flatten())
			#print img + "\t" + str(d)
			output.append((d,img))
		topk = sorted(output, key = lambda x : x[0])[0]
		row.append((v[0],v[1],v[2],v[3],"http://s3.amazonaws.com/my89-frame-annotation/public/images_256/"+topk[1]));
		
#   indices = (-scores).argsort()[:5]
         #   predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
#            meta = [
 #               (p, '%.5f' % scores[i])
 #               for i, p in zip(indices, predictions)
  #          ]
   #         logging.info('result: %s', str(meta))

            # Compute expected information gain
    #        expected_infogain = np.dot(
    #            self.bet['probmat'], scores[self.bet['idmapping']])
    #        expected_infogain *= self.bet['infogain']

            # sort the scores
     #       infogain_sort = expected_infogain.argsort()[::-1]
      #      bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
       #                   for v in infogain_sort[:5]]
           # logging.info('bet result: %s', str(bet_result))

            return (True, "", row, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=8080)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    print "done setup... warm start";
    app.clf.net.forward()
    print "done warm start."

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
