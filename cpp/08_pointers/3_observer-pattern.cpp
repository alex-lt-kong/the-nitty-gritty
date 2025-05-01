#include <memory>
#include <print>
#include <string>
#include <vector>

struct IObserver {
  virtual ~IObserver() = default;
  virtual void update(const std::string &message) = 0;
};

class Subject {
  std::vector<std::weak_ptr<IObserver>> m_observers;

public:
  void attach(const std::shared_ptr<IObserver> &observer) {
    m_observers.push_back(observer);
  }

  // Detach an observer by removing any weak_ptr that locks to it.
  void detach(const std::shared_ptr<IObserver> &observer) {
    std::erase_if(m_observers, [&observer](const std::weak_ptr<IObserver> &wk) {
      return wk.lock() == observer;
    });
  }

  // Notify observers with a dummy update message.
  void notify(const std::string &updateMessage) {
    // Remove expired weak pointers.
    // If wk.expired(), the resource pointed by std::weak_ptr has been released.
    // std::erase_if()'s time complexity for std::vector should be O(n)
    std::erase_if(m_observers, [](const std::weak_ptr<IObserver> &wk) {
      return wk.expired();
    });

    for (const auto &wk : m_observers) {
      if (auto obs = wk.lock()) {
        obs->update(updateMessage);
      }
    }
  }
};

// A concrete observer implementation that prints the update message.
class ObserverImpl : public IObserver {
private:
  int m_id;

public:
  explicit ObserverImpl(const int id) : m_id(id) {}

  void update(const std::string &updateMessage) override {
    std::println("Observer {} received update: {}", m_id, updateMessage);
  }
};

int main() {
  Subject subject;

  auto observer1 = std::make_shared<ObserverImpl>(1);
  auto observer2 = std::make_shared<ObserverImpl>(2);
  auto observer3 = std::make_shared<ObserverImpl>(3);

  subject.attach(observer1);
  subject.attach(observer2);
  subject.attach(observer3);

  {
    auto observer4 = std::make_shared<ObserverImpl>(4);
    subject.attach(observer4);

    std::println("First Notification (with observers 1, 2, 3, 4):");
    subject.notify("Dummy update value - First Update");
  }

  std::println("\nSecond Notification (after observer4 was destroyed):");
  subject.notify("Dummy update value - Second Update");

  // Detach observer1 and then notify.
  std::println("\nDetaching observer 1...");
  subject.detach(observer1);

  std::println("Third Notification (observers 2 and 3):");
  subject.notify("Dummy update value - Third Update");

  return 0;
}
