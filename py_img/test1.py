#python处理图像
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
img=Image.open("C:/Users/29618\Desktop/46448@1589630430@2.png")
'''
print("该图片的格式为：",img.format)
print("该图片的色彩格式是：",img.mode)
print("该图片的宽度高度分别是：",img.size)
img_resize=img.resize((250,250))
img_resize.show()
img_resize.save("img_resize.png")
img_rotate=img.rotate(45)
img_rotate.show()
img_rotate.save("img_rotate.png")
og=img.convert('L')
og.show()
og.save("img_convert.png")
oa=img.filter(ImageFilter.BLUR)
oa.show()
oa.save("img_blur.png")
ob=img.filter(ImageFilter.CONTOUR)
ob.show()
ob.save("img_contour.png")
oc=img.filter(ImageFilter.DETAIL)
oc.show()
oc.save("img_detail.png")
od=img.filter(ImageFilter.EDGE_ENHANCE)
od.show()
od.save("img_edge_enhance.png")
oe=img.filter(ImageFilter.EMBOSS)
oe.show()
oe.save("img_emboss.png")
of=img.filter(ImageFilter.FIND_EDGES)
of.show()
of.save("img_find_edges.png")
oh=img.filter(ImageFilter.SMOOTH)
oh.show()
oh.save("img_smooth.png")
oi=img.filter(ImageFilter.SHARPEN)
oi.show()
oi.save("img_sharpen.png")
om=ImageEnhance.Contrast(img).enhance(20)
om.show()
om.save("contrast.png")
box1=(100,100,400,400)
box2=(400,100,700,400)
region=img.crop(box1)
img.show()
region.show()
region=region.rotate(180)
img.paste(region,box2)
img.show()
r,g,b=img.split()
r.show()
g.show()
b.show()
img.save("mig.jpg")
img2=Image.merge("RGB",(g,b,r))
img2.show()
'''
#拷贝图像
'''
img_copy=Image.new(img.mode,img.size)
a,b=img.size
for i in range(0,a):
    for j in range(0,b):
        x=img.getpixel(a,b)
        img_copy.putpixel((a,b),x)
img_copy.save("img_copy.png")
'''
#写成函数的形式：
'''
def img_copy(im):
    ic=Image.new(im.mode,im.size)
    a,b=im.size
    for i in range(0,a):
        for j in range(0,b):
            x=im.getpixel((i,j))
            ic.putpixel((i,j),x)
    ic.save("img_copy.png")
img_copy(img)
'''
'''
def img_crop(im,box):
    a,b,c,d=box
    ic=Image.new(im.mode,(c-a,d-b))
    for i in range(a,c):
        for j in range(b,d):
            x=im.getpixel((i,j))
            ic.putpixel((i-a,j-c),x)
    ic.save("img_crop_def.png")
    ic.show()
box=(100,100,900,900)
img_crop(img,box)
'''
'''
def img_flip(im):
    b,a=im.size
    imf=Image.new(im.mode,(a,b))
    for i in range(0,a):
        for j in range(0,b):
            x=im.getpixel((j,i))
            imf.putpixel((i,j),x)
    imf.save("img_flip.png")
    imf.show()
img_flip(img)
'''
'''
def img_mirror(im):
    a,b=im.size
    imm=Image.new(im.mode,im.size)
    for i in range(0,a):
        for j in range(0,b):
            x=im.getpixel((i,j))
            imm.putpixel((i,b-j-1),x)
    imm.save("img_mirror2.png")
    imm.show()
img_mirror(img)
'''
'''
def smooth(im,x):
    a,b=x
    w,h=im.size
    pix_R=0;pix_G=0;pix_B=0
    if(b==0 and a==0):
        for i in range(0,2):
            for j in range(0,2):
                pix_R+=im.getpixel((a+i,b+j))[0]
                pix_G+=im.getpixel((a+i,b+j))[1]
                pix_B+=im.getpixel((a+i,b+j))[2]
        return (int(pix_R/4),int(pix_G/4),int(pix_B/4))
    elif(b==0 and a==w-1):
        for i in range(-1,1):
            for j in range(0,2):
                pix_R+=im.getpixel((a+i,b+j))[0]
                pix_G+=im.getpixel((a+i,b+j))[1]
                pix_B+=im.getpixel((a+i,b+j))[2]
        return (int(pix_R/4),int(pix_G/4),int(pix_B/4))
    elif(b==h-1 and a==0):
        for i in range(0,2):
            for j in range(-1,1):
                pix_R+=im.getpixel((a+i,b+j))[0]
                pix_G+=im.getpixel((a+i,b+j))[1]
                pix_B+=im.getpixel((a+i,b+j))[2]
        return (int(pix_R/4),int(pix_G/4),int(pix_B/4))
    elif(b==h-1 and a==w-1):
        for i in range(-1,1):
            for j in range(-1,1):
                pix_R+=im.getpixel((a+i,b+j))[0]
                pix_G+=im.getpixel((a+i,b+j))[1]
                pix_B+=im.getpixel((a+i,b+j))[2]
        return (int(pix_R/4),int(pix_G/4),int(pix_B/4))     
    elif(a==w-1 and b!=0):
        for i in range(-1,1):
            for j in range(-1,2):
                pix_R+=im.getpixel((a+i,b+j))[0]
                pix_G+=im.getpixel((a+i,b+j))[1]
                pix_B+=im.getpixel((a+i,b+j))[2]
        return (int(pix_R/6),int(pix_G/6),int(pix_B/6))
    elif(a==0 and b!=0):
        for i in range(0,2):
            for j in range(-1,2):
                pix_R+=im.getpixel((a+i,b+j))[0]
                pix_G+=im.getpixel((a+i,b+j))[1]
                pix_B+=im.getpixel((a+i,b+j))[2]
        return (int(pix_R/6),int(pix_G/6),int(pix_B/6)) 
    elif(a!=0 and b==0):
        for i in range(-1,2):
            for j in range(0,2):
                pix_R+=im.getpixel((a+i,b+j))[0]
                pix_G+=im.getpixel((a+i,b+j))[1]
                pix_B+=im.getpixel((a+i,b+j))[2]
        return (int(pix_R/6),int(pix_G/6),int(pix_B/6))
    elif(a!=0 and b==h-1):
        for i in range(-1,2):
            for j in range(-1,1):
                pix_R+=im.getpixel((a+i,b+j))[0]
                pix_G+=im.getpixel((a+i,b+j))[1]
                pix_B+=im.getpixel((a+i,b+j))[2]
        return (int(pix_R/6),int(pix_G/6),int(pix_B/6))
    else:
        for i in range(-1,2):
            for j in range(-1,2):
                pix_R+=im.getpixel((a+i,b+j))[0]
                pix_G+=im.getpixel((a+i,b+j))[1]
                pix_B+=im.getpixel((a+i,b+j))[2]
        return (int(pix_R/9),int(pix_G/9),int(pix_B/9))

def img_smooth(im):
    a,b=im.size
    img_smooth=Image.new(im.mode,im.size)
    for i in range(0,a-1):
        for j in range(0,b-1):
            x=smooth(im,(i,j))
            img_smooth.putpixel((i,j),x)
    img_smooth.save("img_smooth2.png")
    img_smooth.show()

img_smooth(img)
'''
