import os
import beyond.Reaper
import time

base_path = 'C:/reaper_auto_reverb/BeyondReaper/'

#add for testifying the functions of beyond


os.system('start D:/Reaper-Win-Portable/reaper.exe') #start是为了让os进程不会中止，可以继续
time.sleep(5)
Reaper.Main_openProject(base_path+'reverb_module.rpp')
time.sleep(5)



def get_media_track(data): #将double类型的指针转换成GetTrack函数的返回形式
    str_data = str(hex(int(data)))
    ptr_data = '(MediaTrack*)0x000000000' + str_data[2:].upper()
    return ptr_data

 
reaper_media_track = Reaper.GetTrack(0, 1) #第二个参数，0是第一个，1是第二个
print(reaper_media_track)

Reaper.SetOnlyTrackSelected(reaper_media_track) #选中

tracks_num = Reaper.GetTrackNumSends(reaper_media_track, 0)
print(tracks_num)

fxchain_file = 'preset_for_reverb/mid_verb.RfxChain'
Reaper.TrackFX_AddByName(reaper_media_track, 'long_verb' + '.RfxChain', False, -1)


# Reaper.Main_OnCommand(number,0)