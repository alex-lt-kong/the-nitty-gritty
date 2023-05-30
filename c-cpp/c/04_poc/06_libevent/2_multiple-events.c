#include <event2/event.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void timer_cb(evutil_socket_t fd, short event, void *arg) {
  const char *msg = (const char *)arg;
  printf("%s\n", msg);
  sleep(strlen(msg) / 10.0);
  printf("%s sleep()ed\n", msg);
}

int main() {
  int retval = 0;
  struct event_base *base = event_base_new();
  if (base == NULL) {
    retval = 1;
    fprintf(stderr, "Failed to create event base\n");
    goto err_event_base_new;
  }

  struct event *timer1 = event_new(base, -1, EV_PERSIST, timer_cb, "Hi");
  if (timer1 == NULL) {
    retval = 1;
    fprintf(stderr, "Failed to event_new() timer1 event\n");
    goto err_event_new_timer1;
  }

  struct event *timer2 =
      event_new(base, -1, EV_PERSIST, timer_cb, "0x0123456789deadbeef");
  if (timer2 == NULL) {
    retval = 1;
    fprintf(stderr, "Failed to event_new() timer2 event\n");
    goto err_event_new_timer2;
  }

  struct timeval tv1, tv2;
  evutil_timerclear(&tv1);
  evutil_timerclear(&tv2);
  tv1.tv_usec = 500 * 1000;
  tv2.tv_sec = 5;

  if (event_add(timer1, &tv1) != 0 || event_add(timer2, &tv2) != 0) {
    retval = 1;
    fprintf(stderr, "Failed to event_add() timers\n");
    goto err_event_add;
  }

  int ret = event_base_dispatch(base);
  if (ret == -1) {
    retval = 1;
    fprintf(stderr, "Failed to event_base_dispatch()\n");
  } else if (ret == 1) {
    printf("event_base_dispatch() returned "
           "as no events were pending or active");
  }

err_event_add:
  event_free(timer2);
err_event_new_timer2:
  event_free(timer1);
err_event_new_timer1:
  event_base_free(base);
err_event_base_new:
  return retval;
}