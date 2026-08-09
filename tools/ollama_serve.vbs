Option Explicit

Dim objShell
Set objShell = CreateObject("WScript.Shell")

' Keep Ollama discoverable through PATH instead of assuming a particular
' Windows username or installation directory.
objShell.Run "ollama serve", 0, False
