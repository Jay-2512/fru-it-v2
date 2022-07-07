from modulefinder import packagePathMap
import cv2 
import numpy as np
from PIL import Image

class Stage3:

    def detect_mango_decay(self, img_array, mango_ripeness):
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (500, 500))

        # cv2.imshow('RGB RAW Image', img)

        # denoising the image
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # convert to hsv
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

        # display HSV image
        # cv2.imshow('HSV Conveted Image', hsv)

        print(f'Mango ripeness is {mango_ripeness}')

        # define lower bound of yellow color
        low_yellow = np.array([12, 100, 100])

        # define high bound of yellow color
        high_yellow = np.array([34,255,255])

        # prepare mask, Mask shows the yellow parts of the mango
        mask = cv2.inRange(hsv, low_yellow, high_yellow)

        # invert mask, Mask to display the decayed parts of the mango
        inv_mask = cv2.bitwise_not(mask)

        # get the number of white pixels from the mask, white pixels are the yellow parts of the mango
        # so this can be used to set the threshold value for the detection

        white_pixels = cv2.countNonZero(mask)
        print('Number of good pixels:', white_pixels)

        # display decay parts, decayed parts will be displayed in cyan and other colors 
        # still have to work on that

        res = cv2.bitwise_not(img, img, mask=inv_mask)

        # display the result
        # cv2.imshow('Decayed Image', res)

        if white_pixels < 60000:
                print('High chances of decay')
                opmsg = 'High chances of decay'
                # bitwise not
                # cv2.imshow('Possible Decay Locations', res)
                rgbres = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgbres)

        elif white_pixels < 65000:
            print('Possible decay')
            opmsg = 'Possible decay'
            # bitwise not
            # cv2.imshow('Possible Decay Locations', res)
            rgbres = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgbres)
        else:
            opmsg = 'No decay'
            print('No decay')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img, opmsg
    
    def detect_banana_decay(self, img_array, banana_ripeness):
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (500, 500))

        # cv2.imshow('RGB RAW Image', img)

        # denoising the image
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # convert to hsv
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

        # display HSV image
        # cv2.imshow('HSV Conveted Image', hsv)

        print(f'Banana ripeness is {banana_ripeness}')

        # define lower bound of yellow color
        low_yellow = np.array([12, 100, 100])

        # define high bound of yellow color
        high_yellow = np.array([34,255,255])

        # prepare mask, Mask shows the yellow parts of the mango
        mask = cv2.inRange(hsv, low_yellow, high_yellow)

        # invert mask, Mask to display the decayed parts of the mango
        inv_mask = cv2.bitwise_not(mask)

        # get the number of white pixels from the mask, white pixels are the yellow parts of the mango
        # so this can be used to set the threshold value for the detection

        white_pixels = cv2.countNonZero(mask)
        print('Number of good pixels:', white_pixels)

        # display decay parts, decayed parts will be displayed in cyan and other colors 
        # still have to work on that

        res = cv2.bitwise_not(img, img, mask=inv_mask)

        # display the result
        # cv2.imshow('Decayed Image', res)

        if white_pixels < 60000:
                print('High chances of decay')
                opmsg = 'High chances of decay'
                # bitwise not
                # cv2.imshow('Possible Decay Locations', res)
                rgbres = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgbres)

        elif white_pixels < 65000:
            print('Possible decay')
            opmsg = 'Possible decay'
            # bitwise not
            # cv2.imshow('Possible Decay Locations', res)
            rgbres = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgbres)
        else:
            opmsg = 'No decay'
            print('No decay')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img, opmsg

    def detect_papaya_decay(self, img_array, papaya_ripeness):
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (500, 500))

        cv2.imshow('RGB RAW Image', img)

        # denoising the image
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # convert to hsv
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

        # display HSV image
        cv2.imshow('HSV Conveted Image', hsv)

        print(f'Papaya ripeness is {papaya_ripeness}')

        # define lower bound of yellow color
        low_yellow = np.array([12, 100, 100])

        # define high bound of yellow color
        high_yellow = np.array([34,255,255])

        # prepare mask, Mask shows the yellow parts of the mango
        mask = cv2.inRange(hsv, low_yellow, high_yellow)

        # invert mask, Mask to display the decayed parts of the mango
        inv_mask = cv2.bitwise_not(mask)

        # get the number of white pixels from the mask, white pixels are the yellow parts of the mango
        # so this can be used to set the threshold value for the detection

        white_pixels = cv2.countNonZero(mask)
        print('Number of good pixels:', white_pixels)

        # display decay parts, decayed parts will be displayed in cyan and other colors 
        # still have to work on that

        res = cv2.bitwise_not(img, img, mask=inv_mask)

        # display the result
        # cv2.imshow('Decayed Image', res)

        if white_pixels < 60000:
                print('High chances of decay')
                opmsg = 'High chances of decay'
                # bitwise not
                # cv2.imshow('Possible Decay Locations', res)
                rgbres = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgbres)

        elif white_pixels < 65000:
            print('Possible decay')
            opmsg = 'Possible decay'
            # bitwise not
            # cv2.imshow('Possible Decay Locations', res)
            rgbres = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgbres)
        else:
            opmsg = 'No decay'
            print('No decay')

        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
        
        return img, opmsg