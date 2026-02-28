// go_parser parses a Go source file and writes a JSON array of semantic units
// to stdout.  Each unit has: unit_type, unit_name (nullable), content, char_offset.
//
// Usage: go run main.go <file.go>
//
// Exit codes:
//   0 — success (even if the file has no extractable units)
//   1 — argument error, I/O error, or parse error
package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
)

type Unit struct {
	UnitType   string  `json:"unit_type"`
	UnitName   *string `json:"unit_name"`
	Content    string  `json:"content"`
	CharOffset int     `json:"char_offset"`
}

func main() {
	// Accept either:
	//   go_parser <file.go>
	//   go_parser -- <file.go>   (used by "go run" to separate source from args)
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: go_parser <file.go>")
		os.Exit(1)
	}

	filename := args[0]
	src, err := os.ReadFile(filename)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filename, src, 0)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	units := extractUnits(fset, f, src)

	// Always emit an array (never null)
	if units == nil {
		units = []Unit{}
	}

	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(units); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func extractUnits(fset *token.FileSet, f *ast.File, src []byte) []Unit {
	var units []Unit

	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			units = append(units, extractFunc(fset, d, src))

		case *ast.GenDecl:
			for _, spec := range d.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				var unitType string
				switch ts.Type.(type) {
				case *ast.StructType:
					unitType = "struct"
				case *ast.InterfaceType:
					unitType = "interface"
				default:
					continue
				}
				units = append(units, extractTypeDecl(fset, d, ts, unitType, src))
			}
		}
	}
	return units
}

func extractFunc(fset *token.FileSet, d *ast.FuncDecl, src []byte) Unit {
	unitType := "function"
	if d.Recv != nil && len(d.Recv.List) > 0 {
		unitType = "method"
	}
	name := d.Name.Name
	start := fset.Position(d.Pos()).Offset
	end := fset.Position(d.End()).Offset
	content := string(src[start:end])
	return Unit{
		UnitType:   unitType,
		UnitName:   &name,
		Content:    content,
		CharOffset: start,
	}
}

func extractTypeDecl(fset *token.FileSet, d *ast.GenDecl, ts *ast.TypeSpec, unitType string, src []byte) Unit {
	name := ts.Name.Name
	// For ungrouped declarations, use the GenDecl extent (includes "type" keyword).
	// For grouped declarations (d.Lparen is valid), use the spec extent.
	var start, end int
	if !d.Lparen.IsValid() {
		start = fset.Position(d.Pos()).Offset
		end = fset.Position(d.End()).Offset
	} else {
		start = fset.Position(ts.Pos()).Offset
		end = fset.Position(ts.End()).Offset
	}
	content := string(src[start:end])
	return Unit{
		UnitType:   unitType,
		UnitName:   &name,
		Content:    content,
		CharOffset: start,
	}
}
