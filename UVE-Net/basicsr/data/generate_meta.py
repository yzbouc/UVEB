import os
path = '/media/xbm/data/VideoDeblur_Dataset/BSD/BSD_3ms24ms_2/train/blur'
f = open('/media/xbm/data/xbm/BasicSR/BasicSR_wave/basicsr/data/meta_info/meta_info_BSD_GT.txt',"w")
filelist = os.listdir(path)
filelist.sort()
for file in filelist:
    path_file = path + '/' + file
    image_list = os.listdir(path_file)
    image_list.sort()
    i = 0
    for image in image_list:
        i += 1
    line = file + ' ' + str(i) + ' ' + '(480,640,3)'
    f.write(line + '\n')
f.close()