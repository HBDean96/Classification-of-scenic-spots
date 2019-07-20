import tkinter
from tkinter import *

from PIL import Image, ImageTk
import tkinter.filedialog
from tkinter import Tk, Label
import predict
from tkinter.messagebox import showinfo
import webbrowser




class AnimatedGIF(Label, object):
    def __init__(self, master, path_to_gif):
        self._master = master
        self._loc = 0

        im = Image.open(path_to_gif)
        self._frames = []
        i = 0
        try:
            while True:
                temp = im.copy()
                self._frames.append(ImageTk.PhotoImage(temp.convert('RGBA')))

                i += 1
                im.seek(i)
        except EOFError:
            pass

        self._len = len(self._frames)

        try:
            self._delay = im.info['duration']
        except:
            self._delay = 100

        self._callback_id = None

        super(AnimatedGIF, self).__init__(master, image=self._frames[0])

    def _run(self):
        self._loc += 1
        if self._loc == self._len:
            self._loc = 0

        self.configure(image=self._frames[self._loc])
        self._callback_id = self._master.after(self._delay, self._run)

    def pack(self, *args, **kwargs):
        self._run()
        super(AnimatedGIF, self).pack(*args, **kwargs)

    def grid(self, *args, **kwargs):
        self._run()
        super(AnimatedGIF, self).grid(*args, **kwargs)

    def place(self, *args, **kwargs):
        self._run()
        super(AnimatedGIF, self).place(*args, **kwargs)

def selectPath():
    global result
    path_ = tkinter.filedialog.askopenfilename(title='打开文件')
    path.set(path_)
    try:
        result = predict.predict_pic(path_)
    except:
        result = 0
    return result


def reply():
    if result == '趵突泉':
        recommendation = 'https://baike.baidu.com/item/%E8%B6%B5%E7%AA%81%E6%B3%89/162170?fr=aladdin'
    elif result == '布达拉宫':
        recommendation = 'https://baike.baidu.com/item/布达拉宫/113399'
    elif result == '鼓浪屿':
        recommendation = 'https://baike.baidu.com/item/鼓浪屿/483700'
    elif result == '莫高窟':
        recommendation = 'https://baike.baidu.com/item/莫高窟/303038'
    elif result == '九寨沟':
        recommendation = 'https://baike.baidu.com/item/九寨沟国家级自然保护区/8762684?fromtitle=九寨沟&fromid=122560'
    elif result == '乐山大佛':
        recommendation = 'https://baike.baidu.com/item/乐山大佛'
    elif result == '丽江古城':
        recommendation = 'https://baike.baidu.com/item/丽江古城'
    elif result == '兵马俑':
        recommendation = 'https://baike.baidu.com/item/兵马俑/60649'
    elif result == '泰山':
        recommendation = 'https://baike.baidu.com/item/泰山/5447'
    elif result == '长城':
        recommendation = 'https://baike.baidu.com/item/长城/14251'



    text.insert(INSERT, "相关网站")

    text.tag_add("link", "1.0", "1.4")
    text.tag_config("link", foreground="blue", underline=True)

    def click(event):
        webbrowser.open(recommendation)

    text.tag_bind("link", "<Button-1>", click)

    showinfo(title='result', message="这幅景点是%s" % result)
    print(result)





if __name__ == "__main__":
    root = Tk()
    root.title("Scenery")
    path = StringVar()
    var = StringVar()
    frame1 = Frame(root)
    frame2 = Frame(root)
    frame3 = Frame(root)


    l = AnimatedGIF(root, "background.gif").pack()

    Label(root, text="请选择您要判断的景点图片", justify=LEFT, font=("幼圆", 20), fg="black").pack()

    Label(frame1, text="目标路径:").pack(side=LEFT)
    Entry(frame1, textvariable=path).pack(side=LEFT, expand=YES, fill=BOTH)
    Button(frame1, text="选择图片", command=selectPath).pack(side=LEFT)

    Button(frame2, text="判断", command=reply, font=("幼圆", 15)).pack(side=TOP)

    frame1.pack(expand=YES, fill=BOTH)
    frame2.pack()

    text = Text(root, width=8, height=1)
    text.pack()
    root.mainloop()


