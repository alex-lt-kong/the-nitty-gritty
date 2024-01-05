using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.Extensions.DependencyModel;

namespace RuntimeCompiler;

public class RuntimeCompiler
{
    public void Compile(string sourceCode, string outputPath, string assemblyName,
        MetadataReference[]? extraReferences = null, OptimizationLevel opt = OptimizationLevel.Release)
    {
        MetadataReference[] defaultReferences = DependencyContext.Default.CompileLibraries
            .SelectMany(cl => cl.ResolveReferencePaths())
            .Select(asm => MetadataReference.CreateFromFile(asm))
            .ToArray();
        var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
        var compilation = CSharpCompilation.Create(assemblyName)
            .WithOptions(new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, optimizationLevel: opt))
            .AddReferences(defaultReferences)
            .AddReferences(extraReferences ?? Array.Empty<MetadataReference>())
            .AddSyntaxTrees(syntaxTree);

        using var stream = new FileStream(outputPath, FileMode.Create);
        var result = compilation.Emit(stream);

        if (result.Success) return;
        var failures = result.Diagnostics
            .Where(diagnostic => diagnostic.IsWarningAsError || diagnostic.Severity == DiagnosticSeverity.Error);

        var errorMessages = string.Join(Environment.NewLine, failures.Select(diagnostic => diagnostic.GetMessage()));
        throw new Exception($"Compilation failed:{Environment.NewLine}{errorMessages}");
    }

    public TInterface CompileWithInterface<TInterface>(string sourceCode, string outputPath, string assemblyName) where TInterface : class
    {
        this.Compile(sourceCode, outputPath, assemblyName,
            new MetadataReference[] { MetadataReference.CreateFromFile(typeof(TInterface).Assembly.Location) });
        var assembly = Assembly.LoadFrom(outputPath);
        //var type = assembly.GetType("MyClass");
        var type = assembly.DefinedTypes.Last();
        var instance = Activator.CreateInstance(type);
        if (instance is not TInterface)
            throw new Exception($"Type {type} does not implement {typeof(TInterface)}");
        return (TInterface)instance;
    }
}