using System.Data.SQLite;
using System.Diagnostics;
using System.Text;

class Student
{
    public int id;
    public string name = "";
    public int score;
    public string partyAffilication = "";
    public string remark = "";
};

class TestClass { 
    static Random rnd = new Random();
    public static Student randomlyGenerateNewStudent()
    {
        string[] party = new string[5] { "CCP", "LDP", "KMT", "DPP", "GOP"};
        Student student = new Student();
        student.name = "Random Student Name" + rnd.Next(65536).ToString();
        student.score = rnd.Next(100);
        student.partyAffilication = party[rnd.Next(5) ];
        student.remark = "student remark" + rnd.Next(65536).ToString();
        return student;
    }
    public static void insertStudents(Student[] students) {
        StringBuilder sql = new StringBuilder("BEGIN TRANSACTION;INSERT INTO 'StudentTable' ('Id', 'Name', 'Score', 'PartyAffilication', 'Remark') VALUES");
        for (int i = 0; i < students.Length; i++)
        {
            sql.Append("(null, '" + students[i].name + "', " + students[i].score.ToString() + ", '" + students[i].partyAffilication + "', " + "'" + students[i].remark + "')");
            if (i < students.Length - 1) sql.Append(",");
            else sql.Append(";COMMIT;");
        }
        using (SQLiteConnection conn = new SQLiteConnection("Data Source=db.sqlite;Version=3;"))
            using (SQLiteCommand cmd = conn.CreateCommand())
            {
                conn.Open();
                cmd.CommandText = sql.ToString();
                cmd.ExecuteNonQuery();
            }
    }
    public static void createEmptyTable() {
        using (SQLiteConnection conn = new SQLiteConnection("Data Source=db.sqlite;Version=3;"))
            using (SQLiteCommand cmd = conn.CreateCommand())
            {
                conn.Open();
                cmd.CommandText = @"DROP TABLE IF EXISTS StudentTable;
                      CREATE TABLE StudentTable(
                      Id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      Name TEXT, 
                      Score TEXT, 
                      PartyAffilication TEXT, 
                      Remark TEXT);";
                cmd.ExecuteNonQuery();
        }
    }

    static void Main(string[] args) {
        TestClass.createEmptyTable();
        int itemCount = 100_000;
        Student[] students = new Student[itemCount];
        for (int i = 0; i < itemCount; i++) {
            students[i] = TestClass.randomlyGenerateNewStudent();
        }
        Stopwatch watch = new Stopwatch();
        watch.Start();
        TestClass.insertStudents(students);
        watch.Stop();
        Console.WriteLine($"{watch.ElapsedMilliseconds}ms");
    }

}