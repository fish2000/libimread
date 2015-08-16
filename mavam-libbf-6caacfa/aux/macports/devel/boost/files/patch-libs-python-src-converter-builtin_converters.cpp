--- libs/python/src/converter/builtin_converters.cpp	(revision 56305)
+++ libs/python/src/converter/builtin_converters.cpp	(revision 71050)
@@ -432,5 +432,8 @@
           {
               int err = PyUnicode_AsWideChar(
-                  (PyUnicodeObject *)intermediate
+#if PY_VERSION_HEX < 0x03020000
+                  (PyUnicodeObject *)
+#endif
+                    intermediate
                 , &result[0]
                 , result.size());
