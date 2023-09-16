
import torch



def img_diff_print(img1, img2, img1_name, img2_name):
    assert len(img1.shape)==len(img2.shape), 'imgtensor shape length must be the same'
    assert img1.shape==img2.shape, 'imgtensor shape must be the same'

    name_len = max(len(img1_name), len(img2_name))
    print(
        '\n'+img1_name.rjust(name_len,' ')+' range R:',round(float(img1[...,0,:,:].min()),3), round(float(img1[...,0,:,:].max()),3), 
        '\n'+img2_name.rjust(name_len,' ')+' range R:',round(float(img2[...,0,:,:].min()),3), round(float(img2[...,0,:,:].max()),3), 
        '\n'+img1_name.rjust(name_len,' ')+' range G:',round(float(img1[...,1,:,:].min()),3), round(float(img1[...,1,:,:].max()),3), 
        '\n'+img2_name.rjust(name_len,' ')+' range G:',round(float(img2[...,1,:,:].min()),3), round(float(img2[...,1,:,:].max()),3), 
        '\n'+img1_name.rjust(name_len,' ')+' range B:',round(float(img1[...,2,:,:].min()),3), round(float(img1[...,2,:,:].max()),3), 
        '\n'+img2_name.rjust(name_len,' ')+' range B:',round(float(img2[...,2,:,:].min()),3), round(float(img2[...,2,:,:].max()),3), 
        '\n'+img1_name.rjust(name_len,' ')+' shape:', img1.shape,
        '\n'+img2_name.rjust(name_len,' ')+' shape:', img2.shape,
    )

if __name__ == '__main__':
    a = torch.rand(3,10,10)
    b = torch.rand(3,10,10)
    
    img_diff_print(a,b,'aaa', 'd1asweq')