#include <event2/event.h>
#include <stdio.h>
#include <string.h>

void timer_cb(evutil_socket_t fd, short event, void *arg) {
  const char *msg = (const char *)arg;
  printf("%s\n", msg);
  sleep(strlen(msg));
  printf("%s sleep()ed\n", msg);
}

int main() {
  struct event_base *base = event_base_new();
  if (base == NULL) {
    fprintf(stderr, "Failed to create event base\n");
    return 1;
  }

  struct event *timer1 = event_new(base, -1, EV_PERSIST, timer_cb, "Hello");
  if (timer1 == NULL) {
    fprintf(stderr, "Failed to create timer event\n");
    return 1;
  }

  struct event *timer2 = event_new(base, -1, EV_PERSIST, timer_cb, "W");
  if (timer2 == NULL) {
    fprintf(stderr, "Failed to create timer event\n");
    return 1;
  }

  struct timeval tv1, tv2;
  evutil_timerclear(&tv1);
  tv1.tv_sec = 5;

  evutil_timerclear(&tv2);
  tv2.tv_sec = 1;

  event_add(timer1, &tv1);
  event_add(timer2, &tv2);

  event_base_dispatch(base);

  event_free(timer1);
  event_free(timer2);
  event_base_free(base);

  return 0;
}