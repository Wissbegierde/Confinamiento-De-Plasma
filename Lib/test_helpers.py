"""
Utilidades compartidas para `test_montecarlo.py`, `test_colisiones.py`, etc.
Evita duplicar colores ANSI, afirmaciones y el resumen final.
"""
import sys

VERDE = "\033[92m"
ROJO = "\033[91m"
RESET = "\033[0m"
NEGRIT = "\033[1m"

_resultados = []  # lista de (nombre, pasó: bool)


def setup_utf8_stdout_win():
    """Evita UnicodeEncodeError en Windows con caracteres acentuados."""
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )


def reset_resultados():
    _resultados.clear()


def ok(nombre: str):
    print(f"  {VERDE}OK{RESET}  {nombre}")
    _resultados.append((nombre, True))


def fallo(nombre: str, motivo: str):
    print(f"  {ROJO}FALLO{RESET}  {nombre}")
    print(f"       -> {motivo}")
    _resultados.append((nombre, False))


def afirmar(cond: bool, nombre: str, detalle: str = ""):
    if cond:
        ok(nombre)
    else:
        fallo(nombre, detalle or "condición falsa")


def imprimir_resumen_final(do_exit: bool = True) -> int:
    total = len(_resultados)
    pasaron = sum(1 for _, r in _resultados if r)
    fallaron = total - pasaron
    print(f"\n  {'=' * 44}")
    if fallaron == 0:
        print(f"  {VERDE}{NEGRIT}Todos los tests pasaron ({pasaron}/{total}){RESET}")
    else:
        print(
            f"  {ROJO}{NEGRIT}{fallaron} test(s) fallaron -- {pasaron}/{total} OK{RESET}"
        )
        print("\n  Tests fallidos:")
        for nombre, r in _resultados:
            if not r:
                print(f"    {ROJO}FALLO{RESET} {nombre}")
    print()
    code = 0 if fallaron == 0 else 1
    if do_exit:
        sys.exit(code)
    return code
