#include <QCoreApplication>
#include <iostream>
#include <sqlite3.h>
#include <string>
#include <time.h>
#include <chrono>

#define ITEMCOUNT 1000000

using namespace std;

class Student {
  public:
    int id;
    string name;
    int score;
    string partyAffilication;
    string remark;
};

Student randomlyGenerateNewStudent() {
  string party[5] = { "CCP", "LDP", "KMT", "DPP", "GOP" };
  Student student;
  student.name = "Random Student Name" + to_string(rand() % 65536);
  student.score = rand() % 100;
  student.partyAffilication = party[rand() % 5];
  student.remark = "student remark" + to_string(rand() % 65536);
  return student;
}

int createEmptyTable(sqlite3 *dbPtr) {
  srand (time(NULL));
  char *errMsg = 0;
  std::string sql = "DROP TABLE IF EXISTS StudentTable;"
              "CREATE TABLE StudentTable("
              "Id INTEGER PRIMARY KEY AUTOINCREMENT, "
              "Name TEXT, "
              "Score TEXT, "
              "PartyAffilication TEXT, "
              "Remark TEXT);";
  int retval = sqlite3_exec(dbPtr, sql.c_str(), 0, 0, &errMsg);
  Student stu = randomlyGenerateNewStudent();
  if (retval != SQLITE_OK) {
    printf("Error: %s\n", sqlite3_errmsg(dbPtr));
    sqlite3_close(dbPtr);
    return 1;
  }
  return 0;
}

int insertStudents(sqlite3 *dbPtr, Student students[], int studentCount) {

  char *errMsg = 0;
  string sql = "BEGIN TRANSACTION;INSERT INTO 'StudentTable' ('Id', 'Name', 'Score', 'PartyAffilication', 'Remark') VALUES";
  // In C++, strings are mutable, so no fancy classes such as StringBuildr are needed.
  int retval = -1;
  for (int i = 0; i < studentCount; i++) {
    sql += "(null, '" + students[i].name + "', " +
            to_string(students[i].score) + ", '" + students[i].partyAffilication +
           "', '" + students[i].remark + "'),";
  }
  sql.pop_back();
  sql += ";COMMIT;";

  retval = sqlite3_exec(dbPtr, sql.c_str(), NULL, 0, &errMsg);
  if (retval != SQLITE_OK) {
      std::cerr << "Error Insert" << errMsg << std::endl;
      sqlite3_free(errMsg);
    }

  return 0;
}

Student students[ITEMCOUNT];
// move this declaration out of main() to avoid stackoverflow...
int main()
{
    sqlite3* dbPtr;
    int iterCount = 25;
    int averageElpasedMs = 0;
    int retval = sqlite3_open(".\\db.sqlite", &dbPtr);

    if (retval != SQLITE_OK) {
      return (retval);
    }
    retval = createEmptyTable(dbPtr);
    for (int i = 0; i < iterCount; i++) {
        for (int j = 0; j < ITEMCOUNT; j++) {
          students[j] = randomlyGenerateNewStudent();
        }

        auto start = std::chrono::high_resolution_clock::now();
        insertStudents(dbPtr, students, ITEMCOUNT);
        auto finish = std::chrono::high_resolution_clock::now();
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish-start);
        cout << i << "-th: " << microseconds.count() / 1000 << "ms" << endl;;
        averageElpasedMs += microseconds.count();
    }
    cout << "Average: " << averageElpasedMs / 1000 / iterCount << "ms" << endl;;
    sqlite3_close(dbPtr);
    return (0);
}
