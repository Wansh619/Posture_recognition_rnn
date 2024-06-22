import cv2
def output_write(frames , outputs):
    idx = 0
    dicti = {"smash" ,"lift", "clear" }
    for id ,  frame in enumerate(frames):

        if id%14!=0:
            
            for j, i in enumerate(outputs[idx]):
                if i==1:
                    cv2.putText(frames[id], dicti[j], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)

        else:
            idx+=1

    return frames                


