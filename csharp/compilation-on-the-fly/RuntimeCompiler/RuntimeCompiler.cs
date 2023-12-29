using Microsoft.Extensions.DependencyModel;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;

namespace RuntimeCompiler;

public static class RuntimeCompiler
{
    public static void Compile(string sourceCode, string outputPath, string assemblyName="MyAssembly")
    {
        var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
        MetadataReference[] _ref = 
            DependencyContext.Default.CompileLibraries
                .SelectMany(cl => cl.ResolveReferencePaths())
                .Select(asm => MetadataReference.CreateFromFile(asm))
                .ToArray();
        var compilation = CSharpCompilation.Create(assemblyName)
            .WithOptions(new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary))
            .AddReferences(_ref)
            .AddSyntaxTrees(syntaxTree);
    
        using var stream = new FileStream(outputPath, FileMode.Create);
        var result = compilation.Emit(stream);

        if (result.Success) return;
        var failures = result.Diagnostics
            .Where(diagnostic => diagnostic.IsWarningAsError || diagnostic.Severity == DiagnosticSeverity.Error);

        var errorMessages = string.Join(Environment.NewLine, failures.Select(diagnostic => diagnostic.GetMessage()));
        throw new Exception($"Compilation failed:{Environment.NewLine}{errorMessages}");
    }
}