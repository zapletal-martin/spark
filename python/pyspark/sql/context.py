#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings
import json
from array import array
from itertools import imap

from py4j.protocol import Py4JError
from py4j.java_collections import MapConverter

from pyspark.rdd import _prepare_for_python_RDD
from pyspark.serializers import AutoBatchedSerializer, PickleSerializer
from pyspark.sql.types import StringType, StructType, _infer_type, _verify_type, \
    _infer_schema, _has_nulltype, _merge_type, _create_converter, _python_to_sql_converter
from pyspark.sql.dataframe import DataFrame

__all__ = ["SQLContext", "HiveContext"]


class SQLContext(object):

    """Main entry point for Spark SQL functionality.

    A SQLContext can be used create L{DataFrame}, register L{DataFrame} as
    tables, execute SQL over tables, cache tables, and read parquet files.
    """

    def __init__(self, sparkContext, sqlContext=None):
        """Create a new SQLContext.

        :param sparkContext: The SparkContext to wrap.
        :param sqlContext: An optional JVM Scala SQLContext. If set, we do not instatiate a new
        SQLContext in the JVM, instead we make all calls to this object.

        >>> from datetime import datetime
        >>> allTypes = sc.parallelize([Row(i=1, s="string", d=1.0, l=1L,
        ...     b=True, list=[1, 2, 3], dict={"s": 0}, row=Row(a=1),
        ...     time=datetime(2014, 8, 1, 14, 1, 5))])
        >>> df = sqlCtx.createDataFrame(allTypes)
        >>> df.registerTempTable("allTypes")
        >>> sqlCtx.sql('select i+1, d+1, not b, list[1], dict["s"], time, row.a '
        ...            'from allTypes where b and i > 0').collect()
        [Row(c0=2, c1=2.0, c2=False, c3=2, c4=0...8, 1, 14, 1, 5), a=1)]
        >>> df.map(lambda x: (x.i, x.s, x.d, x.l, x.b, x.time,
        ...                     x.row.a, x.list)).collect()
        [(1, u'string', 1.0, 1, True, ...(2014, 8, 1, 14, 1, 5), 1, [1, 2, 3])]
        """
        self._sc = sparkContext
        self._jsc = self._sc._jsc
        self._jvm = self._sc._jvm
        self._scala_SQLContext = sqlContext

    @property
    def _ssql_ctx(self):
        """Accessor for the JVM Spark SQL context.

        Subclasses can override this property to provide their own
        JVM Contexts.
        """
        if self._scala_SQLContext is None:
            self._scala_SQLContext = self._jvm.SQLContext(self._jsc.sc())
        return self._scala_SQLContext

    def setConf(self, key, value):
        """Sets the given Spark SQL configuration property.
        """
        self._ssql_ctx.setConf(key, value)

    def getConf(self, key, defaultValue):
        """Returns the value of Spark SQL configuration property for the given key.

        If the key is not set, returns defaultValue.
        """
        return self._ssql_ctx.getConf(key, defaultValue)

    def registerFunction(self, name, f, returnType=StringType()):
        """Registers a lambda function as a UDF so it can be used in SQL statements.

        In addition to a name and the function itself, the return type can be optionally specified.
        When the return type is not given it default to a string and conversion will automatically
        be done.  For any other return type, the produced object must match the specified type.

        >>> sqlCtx.registerFunction("stringLengthString", lambda x: len(x))
        >>> sqlCtx.sql("SELECT stringLengthString('test')").collect()
        [Row(c0=u'4')]
        >>> from pyspark.sql.types import IntegerType
        >>> sqlCtx.registerFunction("stringLengthInt", lambda x: len(x), IntegerType())
        >>> sqlCtx.sql("SELECT stringLengthInt('test')").collect()
        [Row(c0=4)]
        """
        func = lambda _, it: imap(lambda x: f(*x), it)
        ser = AutoBatchedSerializer(PickleSerializer())
        command = (func, None, ser, ser)
        pickled_cmd, bvars, env, includes = _prepare_for_python_RDD(self._sc, command, self)
        self._ssql_ctx.udf().registerPython(name,
                                            bytearray(pickled_cmd),
                                            env,
                                            includes,
                                            self._sc.pythonExec,
                                            bvars,
                                            self._sc._javaAccumulator,
                                            returnType.json())

    def inferSchema(self, rdd, samplingRatio=None):
        """Infer and apply a schema to an RDD of L{Row}.

        ::note:
            Deprecated in 1.3, use :func:`createDataFrame` instead

        When samplingRatio is specified, the schema is inferred by looking
        at the types of each row in the sampled dataset. Otherwise, the
        first 100 rows of the RDD are inspected. Nested collections are
        supported, which can include array, dict, list, Row, tuple,
        namedtuple, or object.

        Each row could be L{pyspark.sql.Row} object or namedtuple or objects.
        Using top level dicts is deprecated, as dict is used to represent Maps.

        If a single column has multiple distinct inferred types, it may cause
        runtime exceptions.

        >>> rdd = sc.parallelize(
        ...     [Row(field1=1, field2="row1"),
        ...      Row(field1=2, field2="row2"),
        ...      Row(field1=3, field2="row3")])
        >>> df = sqlCtx.inferSchema(rdd)
        >>> df.collect()[0]
        Row(field1=1, field2=u'row1')

        >>> NestedRow = Row("f1", "f2")
        >>> nestedRdd1 = sc.parallelize([
        ...     NestedRow(array('i', [1, 2]), {"row1": 1.0}),
        ...     NestedRow(array('i', [2, 3]), {"row2": 2.0})])
        >>> df = sqlCtx.inferSchema(nestedRdd1)
        >>> df.collect()
        [Row(f1=[1, 2], f2={u'row1': 1.0}), ..., f2={u'row2': 2.0})]

        >>> nestedRdd2 = sc.parallelize([
        ...     NestedRow([[1, 2], [2, 3]], [1, 2]),
        ...     NestedRow([[2, 3], [3, 4]], [2, 3])])
        >>> df = sqlCtx.inferSchema(nestedRdd2)
        >>> df.collect()
        [Row(f1=[[1, 2], [2, 3]], f2=[1, 2]), ..., f2=[2, 3])]

        >>> from collections import namedtuple
        >>> CustomRow = namedtuple('CustomRow', 'field1 field2')
        >>> rdd = sc.parallelize(
        ...     [CustomRow(field1=1, field2="row1"),
        ...      CustomRow(field1=2, field2="row2"),
        ...      CustomRow(field1=3, field2="row3")])
        >>> df = sqlCtx.inferSchema(rdd)
        >>> df.collect()[0]
        Row(field1=1, field2=u'row1')
        """

        if isinstance(rdd, DataFrame):
            raise TypeError("Cannot apply schema to DataFrame")

        first = rdd.first()
        if not first:
            raise ValueError("The first row in RDD is empty, "
                             "can not infer schema")
        if type(first) is dict:
            warnings.warn("Using RDD of dict to inferSchema is deprecated,"
                          "please use pyspark.sql.Row instead")

        if samplingRatio is None:
            schema = _infer_schema(first)
            if _has_nulltype(schema):
                for row in rdd.take(100)[1:]:
                    schema = _merge_type(schema, _infer_schema(row))
                    if not _has_nulltype(schema):
                        break
                else:
                    warnings.warn("Some of types cannot be determined by the "
                                  "first 100 rows, please try again with sampling")
        else:
            if samplingRatio < 0.99:
                rdd = rdd.sample(False, float(samplingRatio))
            schema = rdd.map(_infer_schema).reduce(_merge_type)

        converter = _create_converter(schema)
        rdd = rdd.map(converter)
        return self.applySchema(rdd, schema)

    def applySchema(self, rdd, schema):
        """
        Applies the given schema to the given RDD of L{tuple} or L{list}.

        ::note:
            Deprecated in 1.3, use :func:`createDataFrame` instead

        These tuples or lists can contain complex nested structures like
        lists, maps or nested rows.

        The schema should be a StructType.

        It is important that the schema matches the types of the objects
        in each row or exceptions could be thrown at runtime.

        >>> from pyspark.sql.types import *
        >>> rdd2 = sc.parallelize([(1, "row1"), (2, "row2"), (3, "row3")])
        >>> schema = StructType([StructField("field1", IntegerType(), False),
        ...     StructField("field2", StringType(), False)])
        >>> df = sqlCtx.applySchema(rdd2, schema)
        >>> sqlCtx.registerRDDAsTable(df, "table1")
        >>> df2 = sqlCtx.sql("SELECT * from table1")
        >>> df2.collect()
        [Row(field1=1, field2=u'row1'),..., Row(field1=3, field2=u'row3')]

        >>> from datetime import date, datetime
        >>> rdd = sc.parallelize([(127, -128L, -32768, 32767, 2147483647L, 1.0,
        ...     date(2010, 1, 1),
        ...     datetime(2010, 1, 1, 1, 1, 1),
        ...     {"a": 1}, (2,), [1, 2, 3], None)])
        >>> schema = StructType([
        ...     StructField("byte1", ByteType(), False),
        ...     StructField("byte2", ByteType(), False),
        ...     StructField("short1", ShortType(), False),
        ...     StructField("short2", ShortType(), False),
        ...     StructField("int", IntegerType(), False),
        ...     StructField("float", FloatType(), False),
        ...     StructField("date", DateType(), False),
        ...     StructField("time", TimestampType(), False),
        ...     StructField("map",
        ...         MapType(StringType(), IntegerType(), False), False),
        ...     StructField("struct",
        ...         StructType([StructField("b", ShortType(), False)]), False),
        ...     StructField("list", ArrayType(ByteType(), False), False),
        ...     StructField("null", DoubleType(), True)])
        >>> df = sqlCtx.applySchema(rdd, schema)
        >>> results = df.map(
        ...     lambda x: (x.byte1, x.byte2, x.short1, x.short2, x.int, x.float, x.date,
        ...         x.time, x.map["a"], x.struct.b, x.list, x.null))
        >>> results.collect()[0] # doctest: +NORMALIZE_WHITESPACE
        (127, -128, -32768, 32767, 2147483647, 1.0, datetime.date(2010, 1, 1),
             datetime.datetime(2010, 1, 1, 1, 1, 1), 1, 2, [1, 2, 3], None)

        >>> df.registerTempTable("table2")
        >>> sqlCtx.sql(
        ...   "SELECT byte1 - 1 AS byte1, byte2 + 1 AS byte2, " +
        ...     "short1 + 1 AS short1, short2 - 1 AS short2, int - 1 AS int, " +
        ...     "float + 1.5 as float FROM table2").collect()
        [Row(byte1=126, byte2=-127, short1=-32767, short2=32766, int=2147483646, float=2.5)]

        >>> from pyspark.sql.types import _parse_schema_abstract, _infer_schema_type
        >>> rdd = sc.parallelize([(127, -32768, 1.0,
        ...     datetime(2010, 1, 1, 1, 1, 1),
        ...     {"a": 1}, (2,), [1, 2, 3])])
        >>> abstract = "byte short float time map{} struct(b) list[]"
        >>> schema = _parse_schema_abstract(abstract)
        >>> typedSchema = _infer_schema_type(rdd.first(), schema)
        >>> df = sqlCtx.applySchema(rdd, typedSchema)
        >>> df.collect()
        [Row(byte=127, short=-32768, float=1.0, time=..., list=[1, 2, 3])]
        """

        if isinstance(rdd, DataFrame):
            raise TypeError("Cannot apply schema to DataFrame")

        if not isinstance(schema, StructType):
            raise TypeError("schema should be StructType")

        # take the first few rows to verify schema
        rows = rdd.take(10)
        # Row() cannot been deserialized by Pyrolite
        if rows and isinstance(rows[0], tuple) and rows[0].__class__.__name__ == 'Row':
            rdd = rdd.map(tuple)
            rows = rdd.take(10)

        for row in rows:
            _verify_type(row, schema)

        # convert python objects to sql data
        converter = _python_to_sql_converter(schema)
        rdd = rdd.map(converter)

        jrdd = self._jvm.SerDeUtil.toJavaArray(rdd._to_java_object_rdd())
        df = self._ssql_ctx.applySchemaToPythonRDD(jrdd.rdd(), schema.json())
        return DataFrame(df, self)

    def createDataFrame(self, rdd, schema=None, samplingRatio=None):
        """
        Create a DataFrame from an RDD of tuple/list and an optional `schema`.

        `schema` could be :class:`StructType` or a list of column names.

        When `schema` is a list of column names, the type of each column
        will be inferred from `rdd`.

        When `schema` is None, it will try to infer the column name and type
        from `rdd`, which should be an RDD of :class:`Row`, or namedtuple,
        or dict.

        If referring needed, `samplingRatio` is used to determined how many
        rows will be used to do referring. The first row will be used if
        `samplingRatio` is None.

        :param rdd: an RDD of Row or tuple or list or dict
        :param schema: a StructType or list of names of columns
        :param samplingRatio: the sample ratio of rows used for inferring
        :return: a DataFrame

        >>> rdd = sc.parallelize([('Alice', 1)])
        >>> df = sqlCtx.createDataFrame(rdd, ['name', 'age'])
        >>> df.collect()
        [Row(name=u'Alice', age=1)]

        >>> from pyspark.sql import Row
        >>> Person = Row('name', 'age')
        >>> person = rdd.map(lambda r: Person(*r))
        >>> df2 = sqlCtx.createDataFrame(person)
        >>> df2.collect()
        [Row(name=u'Alice', age=1)]

        >>> from pyspark.sql.types import *
        >>> schema = StructType([
        ...    StructField("name", StringType(), True),
        ...    StructField("age", IntegerType(), True)])
        >>> df3 = sqlCtx.createDataFrame(rdd, schema)
        >>> df3.collect()
        [Row(name=u'Alice', age=1)]
        """
        if isinstance(rdd, DataFrame):
            raise TypeError("rdd is already a DataFrame")

        if isinstance(schema, StructType):
            return self.applySchema(rdd, schema)
        else:
            if isinstance(schema, (list, tuple)):
                first = rdd.first()
                if not isinstance(first, (list, tuple)):
                    raise ValueError("each row in `rdd` should be list or tuple")
                row_cls = Row(*schema)
                rdd = rdd.map(lambda r: row_cls(*r))
            return self.inferSchema(rdd, samplingRatio)

    def registerRDDAsTable(self, rdd, tableName):
        """Registers the given RDD as a temporary table in the catalog.

        Temporary tables exist only during the lifetime of this instance of
        SQLContext.

        >>> sqlCtx.registerRDDAsTable(df, "table1")
        """
        if (rdd.__class__ is DataFrame):
            df = rdd._jdf
            self._ssql_ctx.registerRDDAsTable(df, tableName)
        else:
            raise ValueError("Can only register DataFrame as table")

    def parquetFile(self, *paths):
        """Loads a Parquet file, returning the result as a L{DataFrame}.

        >>> import tempfile, shutil
        >>> parquetFile = tempfile.mkdtemp()
        >>> shutil.rmtree(parquetFile)
        >>> df.saveAsParquetFile(parquetFile)
        >>> df2 = sqlCtx.parquetFile(parquetFile)
        >>> sorted(df.collect()) == sorted(df2.collect())
        True
        """
        gateway = self._sc._gateway
        jpath = paths[0]
        jpaths = gateway.new_array(gateway.jvm.java.lang.String, len(paths) - 1)
        for i in range(1, len(paths)):
            jpaths[i] = paths[i]
        jdf = self._ssql_ctx.parquetFile(jpath, jpaths)
        return DataFrame(jdf, self)

    def jsonFile(self, path, schema=None, samplingRatio=1.0):
        """
        Loads a text file storing one JSON object per line as a
        L{DataFrame}.

        If the schema is provided, applies the given schema to this
        JSON dataset.

        Otherwise, it samples the dataset with ratio `samplingRatio` to
        determine the schema.

        >>> import tempfile, shutil
        >>> jsonFile = tempfile.mkdtemp()
        >>> shutil.rmtree(jsonFile)
        >>> ofn = open(jsonFile, 'w')
        >>> for json in jsonStrings:
        ...   print>>ofn, json
        >>> ofn.close()
        >>> df1 = sqlCtx.jsonFile(jsonFile)
        >>> sqlCtx.registerRDDAsTable(df1, "table1")
        >>> df2 = sqlCtx.sql(
        ...   "SELECT field1 AS f1, field2 as f2, field3 as f3, "
        ...   "field6 as f4 from table1")
        >>> for r in df2.collect():
        ...     print r
        Row(f1=1, f2=u'row1', f3=Row(field4=11, field5=None), f4=None)
        Row(f1=2, f2=None, f3=Row(field4=22,..., f4=[Row(field7=u'row2')])
        Row(f1=None, f2=u'row3', f3=Row(field4=33, field5=[]), f4=None)

        >>> df3 = sqlCtx.jsonFile(jsonFile, df1.schema())
        >>> sqlCtx.registerRDDAsTable(df3, "table2")
        >>> df4 = sqlCtx.sql(
        ...   "SELECT field1 AS f1, field2 as f2, field3 as f3, "
        ...   "field6 as f4 from table2")
        >>> for r in df4.collect():
        ...    print r
        Row(f1=1, f2=u'row1', f3=Row(field4=11, field5=None), f4=None)
        Row(f1=2, f2=None, f3=Row(field4=22,..., f4=[Row(field7=u'row2')])
        Row(f1=None, f2=u'row3', f3=Row(field4=33, field5=[]), f4=None)

        >>> from pyspark.sql.types import *
        >>> schema = StructType([
        ...     StructField("field2", StringType(), True),
        ...     StructField("field3",
        ...         StructType([
        ...             StructField("field5",
        ...                 ArrayType(IntegerType(), False), True)]), False)])
        >>> df5 = sqlCtx.jsonFile(jsonFile, schema)
        >>> sqlCtx.registerRDDAsTable(df5, "table3")
        >>> df6 = sqlCtx.sql(
        ...   "SELECT field2 AS f1, field3.field5 as f2, "
        ...   "field3.field5[0] as f3 from table3")
        >>> df6.collect()
        [Row(f1=u'row1', f2=None, f3=None)...Row(f1=u'row3', f2=[], f3=None)]
        """
        if schema is None:
            df = self._ssql_ctx.jsonFile(path, samplingRatio)
        else:
            scala_datatype = self._ssql_ctx.parseDataType(schema.json())
            df = self._ssql_ctx.jsonFile(path, scala_datatype)
        return DataFrame(df, self)

    def jsonRDD(self, rdd, schema=None, samplingRatio=1.0):
        """Loads an RDD storing one JSON object per string as a L{DataFrame}.

        If the schema is provided, applies the given schema to this
        JSON dataset.

        Otherwise, it samples the dataset with ratio `samplingRatio` to
        determine the schema.

        >>> df1 = sqlCtx.jsonRDD(json)
        >>> sqlCtx.registerRDDAsTable(df1, "table1")
        >>> df2 = sqlCtx.sql(
        ...   "SELECT field1 AS f1, field2 as f2, field3 as f3, "
        ...   "field6 as f4 from table1")
        >>> for r in df2.collect():
        ...     print r
        Row(f1=1, f2=u'row1', f3=Row(field4=11, field5=None), f4=None)
        Row(f1=2, f2=None, f3=Row(field4=22..., f4=[Row(field7=u'row2')])
        Row(f1=None, f2=u'row3', f3=Row(field4=33, field5=[]), f4=None)

        >>> df3 = sqlCtx.jsonRDD(json, df1.schema())
        >>> sqlCtx.registerRDDAsTable(df3, "table2")
        >>> df4 = sqlCtx.sql(
        ...   "SELECT field1 AS f1, field2 as f2, field3 as f3, "
        ...   "field6 as f4 from table2")
        >>> for r in df4.collect():
        ...     print r
        Row(f1=1, f2=u'row1', f3=Row(field4=11, field5=None), f4=None)
        Row(f1=2, f2=None, f3=Row(field4=22..., f4=[Row(field7=u'row2')])
        Row(f1=None, f2=u'row3', f3=Row(field4=33, field5=[]), f4=None)

        >>> from pyspark.sql.types import *
        >>> schema = StructType([
        ...     StructField("field2", StringType(), True),
        ...     StructField("field3",
        ...         StructType([
        ...             StructField("field5",
        ...                 ArrayType(IntegerType(), False), True)]), False)])
        >>> df5 = sqlCtx.jsonRDD(json, schema)
        >>> sqlCtx.registerRDDAsTable(df5, "table3")
        >>> df6 = sqlCtx.sql(
        ...   "SELECT field2 AS f1, field3.field5 as f2, "
        ...   "field3.field5[0] as f3 from table3")
        >>> df6.collect()
        [Row(f1=u'row1', f2=None,...Row(f1=u'row3', f2=[], f3=None)]

        >>> sqlCtx.jsonRDD(sc.parallelize(['{}',
        ...         '{"key0": {"key1": "value1"}}'])).collect()
        [Row(key0=None), Row(key0=Row(key1=u'value1'))]
        >>> sqlCtx.jsonRDD(sc.parallelize(['{"key0": null}',
        ...         '{"key0": {"key1": "value1"}}'])).collect()
        [Row(key0=None), Row(key0=Row(key1=u'value1'))]
        """

        def func(iterator):
            for x in iterator:
                if not isinstance(x, basestring):
                    x = unicode(x)
                if isinstance(x, unicode):
                    x = x.encode("utf-8")
                yield x
        keyed = rdd.mapPartitions(func)
        keyed._bypass_serializer = True
        jrdd = keyed._jrdd.map(self._jvm.BytesToString())
        if schema is None:
            df = self._ssql_ctx.jsonRDD(jrdd.rdd(), samplingRatio)
        else:
            scala_datatype = self._ssql_ctx.parseDataType(schema.json())
            df = self._ssql_ctx.jsonRDD(jrdd.rdd(), scala_datatype)
        return DataFrame(df, self)

    def load(self, path=None, source=None, schema=None, **options):
        """Returns the dataset in a data source as a DataFrame.

        The data source is specified by the `source` and a set of `options`.
        If `source` is not specified, the default data source configured by
        spark.sql.sources.default will be used.

        Optionally, a schema can be provided as the schema of the returned DataFrame.
        """
        if path is not None:
            options["path"] = path
        if source is None:
            source = self.getConf("spark.sql.sources.default",
                                  "org.apache.spark.sql.parquet")
        joptions = MapConverter().convert(options,
                                          self._sc._gateway._gateway_client)
        if schema is None:
            df = self._ssql_ctx.load(source, joptions)
        else:
            if not isinstance(schema, StructType):
                raise TypeError("schema should be StructType")
            scala_datatype = self._ssql_ctx.parseDataType(schema.json())
            df = self._ssql_ctx.load(source, scala_datatype, joptions)
        return DataFrame(df, self)

    def createExternalTable(self, tableName, path=None, source=None,
                            schema=None, **options):
        """Creates an external table based on the dataset in a data source.

        It returns the DataFrame associated with the external table.

        The data source is specified by the `source` and a set of `options`.
        If `source` is not specified, the default data source configured by
        spark.sql.sources.default will be used.

        Optionally, a schema can be provided as the schema of the returned DataFrame and
        created external table.
        """
        if path is not None:
            options["path"] = path
        if source is None:
            source = self.getConf("spark.sql.sources.default",
                                  "org.apache.spark.sql.parquet")
        joptions = MapConverter().convert(options,
                                          self._sc._gateway._gateway_client)
        if schema is None:
            df = self._ssql_ctx.createExternalTable(tableName, source, joptions)
        else:
            if not isinstance(schema, StructType):
                raise TypeError("schema should be StructType")
            scala_datatype = self._ssql_ctx.parseDataType(schema.json())
            df = self._ssql_ctx.createExternalTable(tableName, source, scala_datatype,
                                                    joptions)
        return DataFrame(df, self)

    def sql(self, sqlQuery):
        """Return a L{DataFrame} representing the result of the given query.

        >>> sqlCtx.registerRDDAsTable(df, "table1")
        >>> df2 = sqlCtx.sql("SELECT field1 AS f1, field2 as f2 from table1")
        >>> df2.collect()
        [Row(f1=1, f2=u'row1'), Row(f1=2, f2=u'row2'), Row(f1=3, f2=u'row3')]
        """
        return DataFrame(self._ssql_ctx.sql(sqlQuery), self)

    def table(self, tableName):
        """Returns the specified table as a L{DataFrame}.

        >>> sqlCtx.registerRDDAsTable(df, "table1")
        >>> df2 = sqlCtx.table("table1")
        >>> sorted(df.collect()) == sorted(df2.collect())
        True
        """
        return DataFrame(self._ssql_ctx.table(tableName), self)

    def cacheTable(self, tableName):
        """Caches the specified table in-memory."""
        self._ssql_ctx.cacheTable(tableName)

    def uncacheTable(self, tableName):
        """Removes the specified table from the in-memory cache."""
        self._ssql_ctx.uncacheTable(tableName)


class HiveContext(SQLContext):

    """A variant of Spark SQL that integrates with data stored in Hive.

    Configuration for Hive is read from hive-site.xml on the classpath.
    It supports running both SQL and HiveQL commands.
    """

    def __init__(self, sparkContext, hiveContext=None):
        """Create a new HiveContext.

        :param sparkContext: The SparkContext to wrap.
        :param hiveContext: An optional JVM Scala HiveContext. If set, we do not instatiate a new
        HiveContext in the JVM, instead we make all calls to this object.
        """
        SQLContext.__init__(self, sparkContext)

        if hiveContext:
            self._scala_HiveContext = hiveContext

    @property
    def _ssql_ctx(self):
        try:
            if not hasattr(self, '_scala_HiveContext'):
                self._scala_HiveContext = self._get_hive_ctx()
            return self._scala_HiveContext
        except Py4JError as e:
            raise Exception("You must build Spark with Hive. "
                            "Export 'SPARK_HIVE=true' and run "
                            "build/sbt assembly", e)

    def _get_hive_ctx(self):
        return self._jvm.HiveContext(self._jsc.sc())


def _create_row(fields, values):
    row = Row(*values)
    row.__FIELDS__ = fields
    return row


class Row(tuple):

    """
    A row in L{DataFrame}. The fields in it can be accessed like attributes.

    Row can be used to create a row object by using named arguments,
    the fields will be sorted by names.

    >>> row = Row(name="Alice", age=11)
    >>> row
    Row(age=11, name='Alice')
    >>> row.name, row.age
    ('Alice', 11)

    Row also can be used to create another Row like class, then it
    could be used to create Row objects, such as

    >>> Person = Row("name", "age")
    >>> Person
    <Row(name, age)>
    >>> Person("Alice", 11)
    Row(name='Alice', age=11)
    """

    def __new__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Can not use both args "
                             "and kwargs to create Row")
        if args:
            # create row class or objects
            return tuple.__new__(self, args)

        elif kwargs:
            # create row objects
            names = sorted(kwargs.keys())
            values = tuple(kwargs[n] for n in names)
            row = tuple.__new__(self, values)
            row.__FIELDS__ = names
            return row

        else:
            raise ValueError("No args or kwargs")

    def asDict(self):
        """
        Return as an dict
        """
        if not hasattr(self, "__FIELDS__"):
            raise TypeError("Cannot convert a Row class into dict")
        return dict(zip(self.__FIELDS__, self))

    # let obect acs like class
    def __call__(self, *args):
        """create new Row object"""
        return _create_row(self, args)

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        try:
            # it will be slow when it has many fields,
            # but this will not be used in normal cases
            idx = self.__FIELDS__.index(item)
            return self[idx]
        except IndexError:
            raise AttributeError(item)

    def __reduce__(self):
        if hasattr(self, "__FIELDS__"):
            return (_create_row, (self.__FIELDS__, tuple(self)))
        else:
            return tuple.__reduce__(self)

    def __repr__(self):
        if hasattr(self, "__FIELDS__"):
            return "Row(%s)" % ", ".join("%s=%r" % (k, v)
                                         for k, v in zip(self.__FIELDS__, self))
        else:
            return "<Row(%s)>" % ", ".join(self)


def _test():
    import doctest
    from pyspark.context import SparkContext
    from pyspark.sql import Row, SQLContext
    import pyspark.sql.context
    globs = pyspark.sql.context.__dict__.copy()
    sc = SparkContext('local[4]', 'PythonTest')
    globs['sc'] = sc
    globs['sqlCtx'] = sqlCtx = SQLContext(sc)
    globs['rdd'] = rdd = sc.parallelize(
        [Row(field1=1, field2="row1"),
         Row(field1=2, field2="row2"),
         Row(field1=3, field2="row3")]
    )
    globs['df'] = sqlCtx.createDataFrame(rdd)
    jsonStrings = [
        '{"field1": 1, "field2": "row1", "field3":{"field4":11}}',
        '{"field1" : 2, "field3":{"field4":22, "field5": [10, 11]},'
        '"field6":[{"field7": "row2"}]}',
        '{"field1" : null, "field2": "row3", '
        '"field3":{"field4":33, "field5": []}}'
    ]
    globs['jsonStrings'] = jsonStrings
    globs['json'] = sc.parallelize(jsonStrings)
    (failure_count, test_count) = doctest.testmod(
        pyspark.sql.context, globs=globs, optionflags=doctest.ELLIPSIS)
    globs['sc'].stop()
    if failure_count:
        exit(-1)


if __name__ == "__main__":
    _test()
