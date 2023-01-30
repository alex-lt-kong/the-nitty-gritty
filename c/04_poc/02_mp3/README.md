* `apt install libmpg123-dev libao-dev`

* To use a USB sound card (https://stackoverflow.com/questions/44395231/how-to-play-audio-via-a-usb-sound-card-using-c-libao-alsa-on-raspberry-pi):
`~/.asoundrc`
``` 
pcm.!default {
        type asym
        playback.pcm "hw:1,0"
        capture.pcm "hw:1,0"
}

ctl.!default {
        type hw
        card 1
}
```

* `vim /usr/share/alsa/alsa.conf`:
change from
```
defaults.ctl.card 0
defaults.pcm.card 0
```
to
```
vim /usr/share/alsa/alsa.conf
defaults.ctl.card 1
defaults.pcm.card 1
```