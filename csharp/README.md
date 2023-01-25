# C#

* It is possible to use C# without Visual Studio but with VS Code.

* To prepare the development environment, follow VS Code's
[official guide](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.csharp)
and download a few necessarily components, such as
[.NET 6 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/6.0) and
[MSBuild Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)


* To create a new project:
    1. Run `dotnet new sln --name <solution name>` to create a solution file
    (Having a solution is not a must to just compile a project, but C#'s
    Intellisense needs it to function)
    1. Run ` dotnet new console --name <project name>` to create a C#
    project file for a C# console program
    1. Run `dotnet sln add MyProject.csproj` to add the C# project to the
    solution