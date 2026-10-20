//! Python ``repr`` emulation for the values that flow through
//! ``ENode._mk`` (``(op, children, repr(attrs))``) — the canonical
//! member-sort key ``mlopt_core.egraph.types._member_key`` precomputes
//! per enode.
//!
//! Extraction scans each e-class's members in that key's order, and
//! exact ``(total, nops)`` ties resolve to the FIRST member scanned —
//! so the native extractor must reproduce ``repr(attrs)`` byte-for-byte
//! or tie-breaks can diverge from the reference engine.  The covered
//! domain is the ``AttrVal`` alphabet (None / bool / int / float / str
//! / tuple) — the same values ``attrs`` may carry end to end.
//!
//! Limitations (documented, deliberate): CPython's str repr decides
//! printability through the full Unicode database.  Without bundling
//! one, codepoints >= 0x100 are treated as printable except the C1
//! range, NBSP (U+00A0) and SOFT HYPHEN (U+00AD) — exact for ASCII and
//! for every spelling the e-graph produces in practice (``repr`` of
//! ``Var``/``Param``/``Const`` names and attr values).

use crate::attr::AttrVal;
use std::fmt::Write as _;

/// CPython ``repr`` of a string: quote selection + escapes.
///
/// Quotes: ``'`` preferred; ``"`` when the string contains ``'`` but
/// not ``"``.  Escapes: ``\\``, the active quote, ``\t``/``\n``/``\r``,
/// other C0/C1/DEL/NBSP/SOFT-HYPHEN as ``\xXX``; the surrogate range
/// cannot appear in a Rust ``&str`` and astral unprintables degrade to
/// ``\UXXXXXXXX`` only when we *know* they are unprintable — we treat
/// >= 0x100 as printable (see the module docstring).
pub fn repr_str(s: &str) -> String {
    let has_sq = s.contains('\'');
    let has_dq = s.contains('"');
    let quote = if has_sq && !has_dq { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..=0xa0).contains(&(c as u32)) || c as u32 == 0xad => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// CPython ``repr`` of a finite float (shortest round-trip + the
/// ``%r`` formatting rule: scientific iff exp10 < -4 or >= 16).
///
/// Rust's ``{:e}`` yields the same shortest round-trip digits Ryu
/// derives for CPython's ``float_repr`` — the difference is only
/// formatting (exponent style, trailing ``.0``), which this rewrites.
pub fn repr_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // "d[.ddd]e[-]X" — mantissa digits + decimal exponent.
    let sci = format!("{:e}", f);
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp10: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    debug_assert!(!digits.is_empty());
    if (-4..16).contains(&exp10) {
        // Fixed point: place the decimal point exp10+1 digits in.
        let mut out = String::new();
        if neg {
            out.push('-');
        }
        let nd = digits.len() as i32;
        let point = exp10 + 1;
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..(-point) {
                out.push('0');
            }
            out.push_str(&digits);
        } else if point >= nd {
            out.push_str(&digits);
            for _ in 0..(point - nd) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
        out
    } else {
        // Scientific: "d[.ddd]e[+-]XX" — exponent at least two digits.
        let mut out = String::new();
        if neg {
            out.push('-');
        }
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp10 < 0 { '-' } else { '+' });
        let _ = write!(out, "{:02}", exp10.abs());
        out
    }
}

/// CPython ``repr`` of one attr value (the leaf cases of
/// ``repr``-keyed ``_attr_key``).
pub fn repr_attr(v: &AttrVal) -> String {
    match v {
        AttrVal::None => "None".to_string(),
        AttrVal::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        AttrVal::Int(i) => i.to_string(),
        AttrVal::Float(f) => repr_float(*f),
        AttrVal::Str(s) => repr_str(s),
        AttrVal::Tuple(items) => {
            let inner: Vec<String> = items.iter().map(repr_attr).collect();
            match inner.len() {
                0 => "()".to_string(),
                1 => format!("({},)", inner[0]),
                _ => format!("({})", inner.join(", ")),
            }
        }
    }
}

/// CPython ``repr`` of an enode ``attrs`` tuple —
/// ``repr(((k, v), ...))`` — the third element of ``ENode._mk``.
pub fn repr_attrs(attrs: &[(String, AttrVal)]) -> String {
    let inner: Vec<String> = attrs
        .iter()
        .map(|(k, v)| format!("({}, {})", repr_str(k), repr_attr(v)))
        .collect();
    match inner.len() {
        0 => "()".to_string(),
        1 => format!("({},)", inner[0]),
        _ => format!("({})", inner.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_repr_quotes() {
        assert_eq!(repr_str("dim"), "'dim'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("both'\""), "'both\\'\"'");
        assert_eq!(repr_str(""), "''");
    }

    #[test]
    fn str_repr_escapes() {
        assert_eq!(repr_str("a\nb"), "'a\\nb'");
        assert_eq!(repr_str("a\tb\rc"), "'a\\tb\\rc'");
        assert_eq!(repr_str("\x01\x1f\x7f"), "'\\x01\\x1f\\x7f'");
        assert_eq!(repr_str("a\u{a0}b"), "'a\\xa0b'");
        assert_eq!(repr_str("a\u{ad}b"), "'a\\xadb'");
        // Backslash and quote interplay.
        assert_eq!(repr_str("a\\b"), "'a\\\\b'");
    }

    #[test]
    fn float_repr_fixed() {
        assert_eq!(repr_float(1.0), "1.0");
        assert_eq!(repr_float(-0.0), "-0.0");
        assert_eq!(repr_float(0.5), "0.5");
        assert_eq!(repr_float(2.5e-4), "0.00025");
        assert_eq!(repr_float(1e-4), "0.0001");
        assert_eq!(repr_float(1e15), "1000000000000000.0");
        assert_eq!(repr_float(123.456), "123.456");
        assert_eq!(repr_float(-7.25), "-7.25");
    }

    #[test]
    fn float_repr_scientific() {
        assert_eq!(repr_float(1e16), "1e+16");
        assert_eq!(repr_float(1e-5), "1e-05");
        assert_eq!(repr_float(1.5e300), "1.5e+300");
        assert_eq!(repr_float(-2.5e-7), "-2.5e-07");
        assert_eq!(repr_float(f64::INFINITY), "inf");
        assert_eq!(repr_float(f64::NEG_INFINITY), "-inf");
        assert_eq!(repr_float(f64::NAN), "nan");
        assert_eq!(repr_float(1e100), "1e+100");
    }

    #[test]
    fn attrs_repr_shape() {
        assert_eq!(repr_attrs(&[]), "()");
        let one = vec![("dim".to_string(), AttrVal::Int(-1))];
        assert_eq!(repr_attrs(&one), "(('dim', -1),)");
        let two = vec![
            ("dim".to_string(), AttrVal::Int(-1)),
            ("mode".to_string(), AttrVal::Str("x".to_string())),
        ];
        assert_eq!(repr_attrs(&two), "(('dim', -1), ('mode', 'x'))");
    }
}
