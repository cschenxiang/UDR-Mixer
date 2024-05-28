from skimage.measure import compare_ssim, compare_psnr, compare_mse
import cv2
import os



if __name__ == "__main__":
    target ="F:/NeurIPS2024/result/input/"
    input1 = "F:/NeurIPS2024/result/target/"
    mse1 = 0

    for filename in os.listdir(input1):

        a =input1+"/"+filename
        pred = cv2.imread(target+"/"+filename)
        gt = cv2.imread(input1+"/"+filename)
        mse = compare_mse(pred, gt)
        print('MSE：{}'.format(mse))

        mse1 +=mse


    print(mse1)
    print(mse1/500)

