import re
import pyodbc
import sqlalchemy
import contextlib
import sqlalchemy.connectors.pyodbc

class AcNumeric(sqlalchemy.types.Numeric):
	def get_col_spec(self):
		return "NUMERIC"

	def bind_processor(self, dialect):
		return sqlalchemy.processors.to_str

	def result_processor(self, dialect, coltype):
		return None

class AcFloat(sqlalchemy.types.Float):
	def get_col_spec(self):
		return "FLOAT"

	def bind_processor(self, dialect):
		"""By converting to string, we can use Decimal types round-trip."""
		return sqlalchemy.processors.to_str

class AcInteger(sqlalchemy.types.Integer):
	def get_col_spec(self):
		return "INTEGER"

class AcTinyInteger(sqlalchemy.types.Integer):
	def get_col_spec(self):
		return "TINYINT"

class AcSmallInteger(sqlalchemy.types.SmallInteger):
	def get_col_spec(self):
		return "SMALLINT"

class AcDateTime(sqlalchemy.types.DateTime):
	def get_col_spec(self):
		return "DATETIME"

class AcDate(sqlalchemy.types.Date):

	def get_col_spec(self):
		return "DATETIME"

class AcText(sqlalchemy.types.Text):
	def get_col_spec(self):
		return "MEMO"

class AcString(sqlalchemy.types.String):
	def get_col_spec(self):
		return "TEXT" + (self.length and ("(%d)" % self.length) or "")

class AcUnicode(sqlalchemy.types.Unicode):
	def get_col_spec(self):
		return "TEXT" + (self.length and ("(%d)" % self.length) or "")

	def bind_processor(self, dialect):
		return None

	def result_processor(self, dialect, coltype):
		return None

class AcChar(sqlalchemy.types.CHAR):
	def get_col_spec(self):
		return "TEXT" + (self.length and ("(%d)" % self.length) or "")

class AcBinary(sqlalchemy.types.LargeBinary):
	def get_col_spec(self):
		return "BINARY"

class AcBoolean(sqlalchemy.types.Boolean):
	def get_col_spec(self):
		return "YESNO"

class AcTimeStamp(sqlalchemy.types.TIMESTAMP):
	def get_col_spec(self):
		return "TIMESTAMP"

colspecs = {
		sqlalchemy.types.Unicode: AcUnicode,
		sqlalchemy.types.Integer: AcInteger,
		sqlalchemy.types.SmallInteger: AcSmallInteger,
		sqlalchemy.types.Numeric: AcNumeric,
		sqlalchemy.types.Float: AcFloat,
		sqlalchemy.types.DateTime: AcDateTime,
		sqlalchemy.types.Date: AcDate,
		sqlalchemy.types.String: AcString,
		sqlalchemy.types.LargeBinary: AcBinary,
		sqlalchemy.types.Boolean: AcBoolean,
		sqlalchemy.types.Text: AcText,
		sqlalchemy.types.CHAR: AcChar,
		sqlalchemy.types.TIMESTAMP: AcTimeStamp,
	}

#https://docs.microsoft.com/en-us/sql/odbc/reference/appendixes/sql-data-types?view=sql-server-2017
ischema_names = {
	"NUMERIC": sqlalchemy.types.Numeric, 
	"FLOAT": sqlalchemy.types.Float, 
	"INTEGER": sqlalchemy.types.Integer, 
	"TINYINT": sqlalchemy.types.Integer, 
	"SMALLINT": sqlalchemy.types.SmallInteger, 
	"DATETIME": sqlalchemy.types.DateTime, 
	"MEMO": sqlalchemy.types.Text, 
	"TEXT": sqlalchemy.types.String, 
	"BINARY": sqlalchemy.types.LargeBinary, 
	"YESNO": sqlalchemy.types.Boolean, 
	"TIMESTAMP": sqlalchemy.types.TIMESTAMP, 
	"COUNTER": sqlalchemy.types.Integer,
}

class AccessExecutionContext(sqlalchemy.engine.default.DefaultExecutionContext):

	def get_lastrowid(self):
		self.cursor.execute("SELECT @@identity AS lastrowid")
		return self.cursor.fetchone()[0]

class AccessCompiler(sqlalchemy.sql.compiler.SQLCompiler):
	extract_map = sqlalchemy.sql.compiler.SQLCompiler.extract_map.copy()
	extract_map.update({
			'month': 'm',
			'day': 'd',
			'year': 'yyyy',
			'second': 's',
			'hour': 'h',
			'doy': 'y',
			'minute': 'n',
			'quarter': 'q',
			'dow': 'w',
			'week': 'ww'
	})

	def visit_cast(self, cast, **kwargs):
		return cast.clause._compiler_dispatch(self, **kwargs)

	def visit_select_precolumns(self, select):
		"""Access puts TOP, it's version of LIMIT here """
		s = select.distinct and "DISTINCT " or ""
		if select.limit:
			s += "TOP %s " % (select.limit)
		if select.offset:
			raise sqlalchemy.exc.InvalidRequestError(
					'Access does not support LIMIT with an offset')
		return s

	def limit_clause(self, select):
		"""Limit in access is after the select keyword"""
		return ""

	def binary_operator_string(self, binary):
		"""Access uses "mod" instead of "%" """
		return binary.operator == '%' and 'mod' or binary.operator

	function_rewrites = {'current_date': 'now',
						  'current_timestamp': 'now',
						  'length': 'len',
						  }

	def visit_function(self, func, **kwargs):
		"""Access function names differ from the ANSI SQL names;
		rewrite common ones"""
		func.name = self.function_rewrites.get(func.name, func.name)
		return super(AccessCompiler, self).visit_function(func)

	def for_update_clause(self, select):
		"""FOR UPDATE is not supported by Access; silently ignore"""
		return ''

	# Strip schema
	def visit_table(self, table, asfrom=False, **kwargs):
		if asfrom:
			return self.preparer.quote(table.name, table.quote)
		else:
			return ""

	def visit_join(self, join, asfrom=False, **kwargs):
		return ('(' + self.process(join.left, asfrom=True) + \
				(join.isouter and " LEFT OUTER JOIN " or " INNER JOIN ") + \
				self.process(join.right, asfrom=True) + " ON " + \
				self.process(join.onclause) + ')')

	def visit_extract(self, extract, **kw):
		field = self.extract_map.get(extract.field, extract.field)
		return 'DATEPART("%s", %s)' % \
					(field, self.process(extract.expr, **kw))

class AccessDDLCompiler(sqlalchemy.sql.compiler.DDLCompiler):
	def get_column_specification(self, column, **kwargs):
		if column.table is None:
			raise sqlalchemy.exc.CompileError(
							"access requires Table-bound columns "
							"in order to generate DDL")

		colspec = self.preparer.format_column(column)
		seq_col = column.table._autoincrement_column
		if seq_col is column:
			colspec += " AUTOINCREMENT"
		else:
			colspec += " " + self.dialect.type_compiler.process(column.type)

			if column.nullable is not None and not column.primary_key:
				if not column.nullable or column.primary_key:
					colspec += " NOT NULL"
				else:
					colspec += " NULL"

			default = self.get_column_default_string(column)
			if default is not None:
				colspec += " DEFAULT " + default

		return colspec

	def visit_drop_index(self, drop):
		index = drop.element
		self.append("\nDROP INDEX [%s].[%s]" % \
						(index.table.name,
						self._index_identifier(index.name)))

class AccessIdentifierPreparer(sqlalchemy.sql.compiler.IdentifierPreparer):
	reserved_words = sqlalchemy.sql.compiler.RESERVED_WORDS.copy()
	reserved_words.update(['value', 'text'])
	def __init__(self, dialect):
		super(AccessIdentifierPreparer, self).\
				__init__(dialect, initial_quote='[', final_quote=']')

class AccessDialect(sqlalchemy.engine.default.DefaultDialect):
	
	name = 'access'

	_pyodbc_connector = sqlalchemy.connectors.pyodbc.PyODBCConnector
	default_paramstyle = _pyodbc_connector.default_paramstyle
	supports_unicode_binds = _pyodbc_connector.supports_unicode_binds
	supports_native_decimal = _pyodbc_connector.supports_native_decimal
	supports_unicode_statements = _pyodbc_connector.supports_unicode_statements
	supports_sane_multi_rowcount = _pyodbc_connector.supports_sane_multi_rowcount
	supports_sane_rowcount_returning = _pyodbc_connector.supports_sane_rowcount_returning
	del _pyodbc_connector
	
	supports_sane_rowcount = False

	poolclass = sqlalchemy.pool.SingletonThreadPool
	statement_compiler = AccessCompiler
	ddl_compiler = AccessDDLCompiler
	preparer = AccessIdentifierPreparer
	execution_ctx_cls = AccessExecutionContext

	colspecs = colspecs
	ischema_names = ischema_names

	def __init__(self, paramstyle = default_paramstyle, **kwargs):
		super().__init__(paramstyle = paramstyle, **kwargs)

	@classmethod
	def dbapi(cls):
		import pyodbc_hook as module
		return module

	def create_connect_args(self, url):
		# return self.dbapi.create_connect_args(url)

		driverList = tuple(item for item in pyodbc.drivers() if ("Microsoft Access Driver" in item))
		if (not driverList):
			errorMessage = "You need to install 'Microsoft Access Database Engine 2010 Redistributable'. It can be found at: https://www.microsoft.com/en-US/download/details.aspx?id=13255"
			raise SyntaxError(errorMessage)

		if ("Microsoft Access Driver (*.mdb, *.accdb)" in driverList):
			driver = "Microsoft Access Driver (*.mdb, *.accdb)"
		elif (".accdb" in fileName):
			errorMessage = "You need to install 'Microsoft Access Database Engine 2010 Redistributable'. It can be found at: https://www.microsoft.com/en-US/download/details.aspx?id=13255"
			raise SyntaxError(errorMessage)
		else:
			driver = "Microsoft Access Driver (*.mdb)"

		options = url.translate_connect_args()

		connect_args = []
		connect_kwargs = {}

		connect_kwargs["driver"] = driver
		connect_kwargs["dbq"] = options["database"]
		user = options.get("username", None)
		if user:
			connect_kwargs["uid"] = user
			connect_kwargs["pwd"] = options.get('password', '')

		return connect_args, connect_kwargs

	def on_connect(self):
		def connect(connection):
			#Use: https://github.com/mkleehammer/pyodbc/wiki/Unicode
			connection.setdecoding(pyodbc.SQL_CHAR, encoding = "utf-8")
			connection.setdecoding(pyodbc.SQL_WCHAR, encoding = "utf-8")
			connection.setencoding(encoding = "utf-8")
		return connect

	def last_inserted_ids(self):
		return self.context.last_inserted_ids

	def has_table(self, base_connection, tablename, schema = None):
		return bool(tablename in self.get_table_names(base_connection, schema = schema))

	@sqlalchemy.engine.reflection.cache
	def get_table_names(self, base_connection, schema = None, **kwargs):
		"""Returns a list of table names."""

		cursor = base_connection.connection.cursor()

		return tuple(table_info.table_name for tableType in ("TABLE", "ALIAS", "SYNONYM") for table_info in cursor.tables(tableType = tableType))

	@sqlalchemy.engine.reflection.cache
	def get_pk_constraint(self, base_connection, relation, schema = None, **kwargs):
		"""Returns a list of primary keys.
		Modified code from sqlalchemy.dialects.sqlite.base.py
		"""

		constraint_name = None
		keyList = tuple(column["name"] for column in self.get_columns(base_connection, relation, schema, **kwargs) if column['primary_key'])
		return {'constrained_columns': keyList, 'name': constraint_name}

	@sqlalchemy.engine.reflection.cache
	def get_foreign_keys(self, base_connection, relation, schema = None, **kwargs):
		"""MS access does not support foreign keys."""

		return ()

	@sqlalchemy.engine.reflection.cache
	def get_indexes(self, base_connection, relation, schema = None, **kwargs):
		"""Returns a list of indexes.
		Modified code from sqlalchemy.dialects.sqlite.base.py
		"""

		index_info = {}
		cursor = base_connection.connection.cursor()
		for item in cursor.statistics(relation):
			name = item[5]
			if (name is None):
				continue

			if (name in index_info):
				index_info[name]["column_names"].append(item[8])
				continue
				
			index_info[name] = {
				"name": name,
				"column_names": [item[8]],
				"unique": not item[3],
			}

		return tuple(index_info.values())

	@sqlalchemy.engine.reflection.cache
	def get_columns(self, base_connection, relation, schema = None, **kwargs):
		"""Returns a list of dictionaries containing information about each column.
		Modified code from sqlalchemy.dialects.sqlite.base.py
		"""

		column_info = {}
		cursor = base_connection.connection.cursor()
		for item in cursor.columns(table = relation):
			name = item[3]
			column_info[name] = {
				"name": name,
				"type": self._resolve_type_affinity(item[5].upper()),
				"nullable": item[12],
				"primary_key": False,
			}

			default = item[12]
			if (default is None):
				column_info[name]["default"] = None
			else:
				column_info[name]["default"] = sqlalchemy.engine.util.text_type(default)

		for item in cursor.statistics(relation):
			name = item[8]
			if (name is None):
				continue
			column_info[name]["primary_key"] = item[5] is "PrimaryKey"

		return tuple(column_info.values())

	def _resolve_type_affinity(self, type_):
		"""Return a data type from a reflected column, using affinity tules.

		SQLite's goal for universal compatibility introduces some complexity
		during reflection, as a column's defined type might not actually be a
		type that SQLite understands - or indeed, my not be defined *at all*.
		Internally, SQLite handles this with a 'data type affinity' for each
		column definition, mapping to one of 'TEXT', 'NUMERIC', 'INTEGER',
		'REAL', or 'NONE' (raw bits). The algorithm that determines this is
		listed in http://www.sqlite.org/datatype3.html section 2.1.

		This method allows SQLAlchemy to support that algorithm, while still
		providing access to smarter reflection utilities by regcognizing
		column definitions that SQLite only supports through affinity (like
		DATE and DOUBLE).
		"""

		match = re.match(r'([\w ]+)(\(.*?\))?', type_)
		if match:
			coltype = match.group(1)
			args = match.group(2)
		else:
			coltype = ''
			args = ''

		if coltype in self.ischema_names:
			coltype = self.ischema_names[coltype]
		else:
			print("@_resolve_type_affinity", type_)
			gjhhgjjhghg
		# elif 'INT' in coltype:
		# 	coltype = sqltypes.INTEGER
		# elif 'CHAR' in coltype or 'CLOB' in coltype or 'TEXT' in coltype:
		# 	coltype = sqltypes.TEXT
		# elif 'BLOB' in coltype or not coltype:
		# 	coltype = sqltypes.NullType
		# elif 'REAL' in coltype or 'FLOA' in coltype or 'DOUB' in coltype:
		# 	coltype = sqltypes.REAL
		# else:
		# 	coltype = sqltypes.NUMERIC

		if args is not None:
			args = re.findall(r'(\d+)', args)
			try:
				coltype = coltype(*[int(a) for a in args])
			except TypeError:
				util.warn(
					"Could not instantiate type %s with "
					"reflected arguments %s; using no arguments." %
					(coltype, args))
				coltype = coltype()
		else:
			coltype = coltype()

		return coltype

	def do_execute(self, cursor, statement, params, context = None, **kwargs):
		"""Sets up the execute statement.
		Modified code from: https://stackoverflow.com/questions/9233912/connecting-sqlalchemy-to-msaccess/13849359#13849359
		"""

		# print("@1", cursor, statement, params, context, kwargs)
		if params == {}:
			params = ()
		super().do_execute(cursor, statement, params, **kwargs)


if __name__ == '__main__':
	import os
	import sys

	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	
	def test():
		moduleLocation = "forks.sqlalchemy.dialects.access.base"

		sqlalchemy.dialects.registry.register("access.fixed", "access.base", "AccessDialect")
		engine = sqlalchemy.create_engine("access+fixed:///./test.accdb")
		connection = engine.connect()

		metadata = sqlalchemy.MetaData(engine, reflect = True)
		print(metadata)

	def raw():
		import pyodbc
		connection = pyodbc.connect(driver = "Microsoft Access Driver (*.mdb, *.accdb)", dbq = "./test.accdb")

		#Use: https://github.com/mkleehammer/pyodbc/wiki/Unicode
		connection.setdecoding(pyodbc.SQL_CHAR, encoding = "utf-8")
		connection.setdecoding(pyodbc.SQL_WCHAR, encoding = "utf-8")
		connection.setencoding(encoding = "utf-8")

		print(connection)
		cursor = connection.cursor()
		tableNames = [x[2] for x in cursor.tables().fetchall() if x[3] == 'TABLE']
		print(tableNames)

	test()
	# raw()