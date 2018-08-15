import os
import cv2
import numpy as np
import histomicstk as htk
import numpy as np
import scipy as sp
import skimage.io
import skimage.measure
import skimage.color
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
plt.rcParams['figure.figsize'] = 15, 15
plt.rcParams['image.cmap'] = 'gray'
titlesize = 24


def save_normalized_images(inputImageFile, refImageFile, save_path):
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    name = inputImageFile.split('/')[-1].split('.')[0]
    imReference = skimage.io.imread(refImageFile)[:, :, :3]
    # get mean and stddev of reference image in lab space
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
    # perform reinhard color normalization
    imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)
    skimage.io.imsave(save_path + name + '.png', imNmzd)
    image = Image.open(save_path + name + '.png')
    return image


def cell_segment_evaluate(intensity, refImageFile, segmenval_original_path, segmenval_label_path):
    totallabel  =0
    totalsegment  = 0
    totalright  = 0
    root_dir = segmenval_original_path
    imList = os.listdir(root_dir)
    for imdata in range(0,len(imList)):

        inputImageFile = (segmenval_original_path + imList[imdata])
        #inputImageFile = ('/disk1/cell_work/data/new/20171005091629.jpg')  # Easy1.png
        name =  imList[imdata].strip('.png')
        imInput = skimage.io.imread(inputImageFile)[:, :, :3]
        imReference = skimage.io.imread(refImageFile)[:, :, :3]
        # get mean and stddev of reference image in lab space
        meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
        # perform reinhard color normalization
        imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)

        w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(imNmzd,I_0=255 )
        I_0=255
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
        # specify stains of input image
        stains = ['hematoxylin',  # nuclei stain
                  'eosin',        # cytoplasm stain
                  'null']    
        # Perform color deconvolution
        deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, w_est, I_0)

        imNucleiStain = deconv_result.Stains[:, :, 1]
        #plt.figure()
        #plt.imshow(imNucleiStain)
        #imNucleiStain =imDeconvolved[:, :, 0]
        foreground_threshold = intensity

        imFgndMask = sp.ndimage.morphology.binary_fill_holes(
            imNucleiStain < foreground_threshold)

        min_radius = 5
        max_radius = 30
        imLog = htk.filters.shape.clog(imNucleiStain, imFgndMask,
                                       sigma_min=min_radius * np.sqrt(2),
                                       sigma_max=max_radius * np.sqrt(2))
        # detect and segment nuclei using local maximum clustering
        local_max_search_radius = 10
        imNucleiSegMask1, Seeds, Max = htk.segmentation.nuclear.max_clustering(
            imLog[0], imFgndMask, local_max_search_radius)
        # filter out small objects
        min_nucleus_area = 200
        imNucleiSegMask = htk.segmentation.label.area_open(
            imNucleiSegMask1, min_nucleus_area).astype(np.int)
        # compute nuclei properties
        objProps = skimage.measure.regionprops(imNucleiSegMask)
       # print 'Number of nuclei = ', len(objProps)

        # prepare groundtruth
        root_data= segmenval_label_path + name+'/'
        imageList = os.listdir(root_data)

        labelist = []

        for i in imageList[0:len(imageList)]:
            img = cv2.imread(root_data + i)
            img = 255-img
            im = img[:,:,1]
            im.reshape(200,200)
            labelist.append(im)

        imNucleicompact = htk.segmentation.label.compact(imNucleiSegMask, compaction=3)

        k= (imNucleicompact==-1)
        imNucleicompact1=np.copy(k)
        plt.rcParams['figure.figsize'] = 15,15    
        for ii in range(0,k.shape[0]):
            for jj in range(0,k.shape[1]):
                if imNucleicompact[ii,jj]>0:
                    imNucleicompact1[ii,jj]=1

        imNucleicompact2 = skimage.measure.label(imNucleicompact1,connectivity = 1)

        right = 0
        segment = 0
        label = 0
        for i in range(1,imNucleicompact2.max()):

            k =  (imNucleicompact2==i)
            location = np.where(k == 1)
            x_min, y_min = min(location[0]),min(location[1])
            x_max, y_max = max(location[0]),max(location[1])
            space = (x_max-x_min)*(y_max-y_min)

            if space<450 and space>100:

                for im in labelist:
                    result = k*im
                    if  result.sum()>255*100:
                        right= result.sum()/255 + right
                        segment = k.sum() + segment

            if space>449:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7)) 
                k.dtype=np.uint8
                eroded=cv2.erode(k,kernel);
                dilated = cv2.dilate(eroded,kernel)
                new_seg = skimage.measure.label(dilated,connectivity = 1)
                for j in range (1,new_seg.max()+1):

                    kk =  (new_seg==j)
                    location1 = np.where(kk == 1)
                    x_min1, y_min1 = min(location1[0]),min(location1[1])
                    x_max1, y_max1 = max(location1[0]),max(location1[1])
                    space1 = (x_max1-x_min1)*(y_max1-y_min1)
                    if space1< 800:
                        for im in labelist:
                            result = kk*im
                            if  result.sum()>255*100:
                                right= result.sum()/255 + right
                                segment = kk.sum() + segment
                                
    # calculate the number of pixel in ground truth, segmentation result and overlapping region
        label= 0
        for im in labelist:
            label = label+ im.sum()/255

        #print (label)              
        #print(segment)
        #print(right)
        totallabel  =label+totallabel
        totalsegment  = segment+totalsegment
        totalright  = right+totalright

    a=totallabel
    b=totalsegment
    c=totalright
    # calculate f-score
    recall = c/a
    precision = c/float(b)
    Fscore=(2*precision*recall)/(precision+recall)
    print('recall,precision:')
    print(recall,precision)
    print('Fscore:')
    print(Fscore)


def cell_segment(image_path, data_saved_path, ref_path, intensity):
    totallabel  =0
    totalsegment  = 0
    totalright  = 0

    plt.rcParams['figure.figsize'] = 15, 15
    plt.rcParams['image.cmap'] = 'gray'
    titlesize = 24

    classification1 = []
    totalseg = []
    name = image_path.split('/')[-1].split('/')[-1].split('.')[0]
    inputImageFile = image_path
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    refImageFile = ref_path
    imReference = skimage.io.imread(refImageFile)[:, :, :3]
    # get mean and stddev of reference image in lab space
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
    # perform reinhard color normalization
    imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)

    w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(imNmzd,I_0=255 )
    I_0=255
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']    
    # Perform color deconvolution
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, w_est, I_0)

    imNucleiStain = deconv_result.Stains[:, :, 1]
    foreground_threshold = intensity

    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
        imNucleiStain < foreground_threshold)

    min_radius = 5
    max_radius = 30
    imLog = htk.filters.shape.clog(imNucleiStain, imFgndMask,
                                   sigma_min=min_radius * np.sqrt(2),
                                   sigma_max=max_radius * np.sqrt(2))
    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 10
    imNucleiSegMask1, Seeds, Max = htk.segmentation.nuclear.max_clustering(
        imLog[0], imFgndMask, local_max_search_radius)
    # filter out small objects
    min_nucleus_area = 200
    imNucleiSegMask = htk.segmentation.label.area_open(
        imNucleiSegMask1, min_nucleus_area).astype(np.int)
    # compute nuclei properties
    #objProps = skimage.measure.regionprops(imNucleiSegMask)
    print ('step one done')

    imNucleicompact = htk.segmentation.label.compact(imNucleiSegMask, compaction=3)
    k= (imNucleicompact==-1)
    imNucleicompact1=np.copy(k)
    plt.rcParams['figure.figsize'] = 15,15    
    for ii in range(0,imNucleicompact.shape[0]):
        for jj in range(0,imNucleicompact.shape[1]):
            if imNucleicompact[ii,jj]>0:
                imNucleicompact1[ii,jj]=1

    imNucleicompact2 = skimage.measure.label(imNucleicompact1,connectivity = 1)
    imInput2 = np.copy(imNmzd)
    plt.rcParams['figure.figsize'] = 1, 1

############## save image and calculate f-score #########
    listt = []
    seglis = []
    list_nuclei = []
    right = 0
    segment = 0
    label = 0
    for i in range(1,imNucleicompact2.max()):

        k =  (imNucleicompact2==i)
        location = np.where(k == 1)
        x_min, y_min = min(location[0]),min(location[1])
        x_max, y_max = max(location[0]),max(location[1])
        space = (x_max-x_min)*(y_max-y_min)

        if space<450 and space>100:           
            segmentate = k[x_min:x_max,y_min:y_max]
            segmentate = np.tile(np.expand_dims(segmentate,axis=2),(1,1,3))
            listt.append([x_min,y_min,x_max,y_max])
            seglis.append(segmentate)
            img1 = imNmzd[x_min:x_max,y_min:y_max,:]
            img1 = img1*segmentate
            list_nuclei.append(img1)
            #plt.imshow(img1)
            #plt.show()

        if space>449:
            #print i
            #print space
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7)) 
            k.dtype=np.uint8
            eroded=cv2.erode(k,kernel);
            dilated = cv2.dilate(eroded,kernel)
            new_seg = skimage.measure.label(dilated,connectivity = 1)
            for j in range (1,new_seg.max()+1):
                #print 'j=',j
                kk =  (new_seg==j)
                location1 = np.where(kk == 1)
                x_min1, y_min1 = min(location1[0]),min(location1[1])
                x_max1, y_max1 = max(location1[0]),max(location1[1])
                space1 = (x_max1-x_min1)*(y_max1-y_min1)
                if space1< 800:
                    segmentate = kk[x_min1:x_max1,y_min1:y_max1]
                    segmentate = np.tile(np.expand_dims(segmentate,axis=2),(1,1,3))
                    listt.append([x_min1,y_min1,x_max1,y_max1])
                    seglis.append(segmentate)
                    img1 = imNmzd[x_min1:x_max1,y_min1:y_max1,:]
                    img1 = img1*segmentate
                    list_nuclei.append(img1)


    #save cell image filled with [255,255,255]
    image_dict = {}
    n=0
    #preparing cell images
    for img in list_nuclei:
        color_mean = img.mean(axis=2)
        for p in range(0, color_mean.shape[0]):
            for q in range(0, color_mean.shape[1]):
                if color_mean[p, q] == 0.0:
                    img[p, q, :] = 255

        height, width = img.shape[0], img.shape[1]
        if max(height,width)>32:
            scale = 32/float(max(height,width))
            height, width = int(height*scale), int(width*scale)
            img = sp.misc.imresize(img, (height, width))

        npad = ((16-height/2,32-height-(16-height/2)),(16-width/2,32-width-(16-width/2)),(0,0))
        segmentate_image = np.pad(img, pad_width=npad,constant_values=255,mode='constant')
        image_dict[n] = segmentate_image
        n+=1

    image = np.array(image_dict.values())
    np.save(((data_saved_path + name)+'.npy'), image)
    print ('Number of nuclei = ', len(image_dict))
    print ('image saved')