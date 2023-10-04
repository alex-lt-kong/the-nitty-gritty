#include <event2/event.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void timer_cb(evutil_socket_t fd, short event, void *arg) {
  const char *msg = (const char *)arg;
  printf("%s\n", msg);
}

int main() {
  int retval = 0;
  struct event_base *base = event_base_new();
  if (base == NULL) {
    retval = 1;
    fprintf(stderr, "Failed to create event base\n");
    goto err_event_base_new;
  }

  struct event *timer =
      event_new(base, -1, EV_PERSIST, timer_cb, "Hello world!");
  if (timer == NULL) {
    retval = 1;
    fprintf(stderr, "Failed to event_new() a timer event\n");
    goto err_event_new;
  }

  struct timeval tv;
  evutil_timerclear(&tv);
  tv.tv_usec = 500 * 1000;

  if (event_add(timer, &tv) != 0) {
    retval = 1;
    fprintf(stderr, "Failed to event_add() timer\n");
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
  event_free(timer);
err_event_new:
  event_base_free(base);
err_event_base_new:
  return retval;
}